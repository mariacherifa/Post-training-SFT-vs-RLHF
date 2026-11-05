"""Two experiments, one evaluation protocol:

1. SFT vs PPO: response-only cross-entropy on the SAME train/validation examples.
2. PPO KL sweep: reward-model score and sampled KL-to-SFT on held-out prompts.

Run from the repository root: python -m scripts.simple_experiments --train-ppo
Or re-evaluate saved PPO checkpoints: python -m scripts.simple_experiments

This script deliberately does NOT compare SFT CE with PPO's clipped policy/value loss;
those optimize different objectives and cannot be compared numerically.
"""

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tinygpt.model import TinyGPT
from src.data.utils import get_full_text
from src.models.reward_model import RewardModel
from scripts.prepare_experiments import load_json 
from tinygpt.data import encode 

EXPERIMENTS = Path("experiments")
DATA = Path("data/shp_processed")


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_llm(path, expected_stoi, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["stoi"] != expected_stoi:
        raise ValueError(f"Different vocabulary/token IDs: {path}")
    config = dict(checkpoint["config"])
    config["num_layers"] = config.pop("num_layer", config.get("num_layers"))
    config = {k: config[k] for k in (
        "n_embd", "vocab_size", "num_heads", "block_size", "num_layers", "dropout"
    )}
    model = TinyGPT(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, config


def load_reward(path, expected_stoi, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("stoi") != expected_stoi:
        raise ValueError("Reward model's vocabulary must match the SFT policy")
    config = checkpoint["config"]
    n_layers = config.get("num_layer", config.get("num_layers"))
    model = RewardModel(
        config["n_embd"], config["num_heads"], config["block_size"],
        config["vocab_size"], config["dropout"], n_layers
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, config["block_size"]


def formatted(instruction, response):
    return get_full_text({"instruction": instruction, "response": response}, task="SFT")


def choose_examples(path, stoi, context, generation_length, limit, seed, train_keys=None):
    """Select deterministic compatible preference examples without prompt duplication."""
    examples = load_json(path)
    rng = random.Random(seed)
    rng.shuffle(examples)
    chosen, seen_instructions, seen_posts = [], set(), set()
    for example in examples:
        instruction = example["instruction"]
        prompt = formatted(instruction, "")
        full = formatted(instruction, example["accepted"])
        if not full.startswith(prompt):
            raise ValueError("SFT formatting: prompt is not a prefix of full accepted text")
        post_id = str(example["post_id"]) if example.get("post_id") is not None else None
        key = (instruction, post_id)
        overlaps_train = train_keys is not None and (
            ("instruction", instruction) in train_keys or
            (post_id is not None and ("post_id", post_id) in train_keys)
        )
        if (instruction in seen_instructions or
                (post_id is not None and post_id in seen_posts) or overlaps_train):
            continue
        # Need room for either the chosen response or sampled responses.
        if not (1 <= len(prompt) < len(full) <= context and
                len(prompt) + generation_length <= context):
            continue
        if any(ch not in stoi for ch in full):
            continue
        chosen.append((prompt, full, key))
        seen_instructions.add(instruction)
        if post_id is not None:
            seen_posts.add(post_id)
        if limit > 0 and len(chosen) >= limit:
            break
    if not chosen:
        raise ValueError(f"No compatible examples available in {path}")
    return chosen


@torch.no_grad()
def response_log_probs(model, ids, prompt_len):
    """Return one log probability for each token after the prompt."""
    if not 1 <= prompt_len < ids.shape[1]:
        raise ValueError("Expected a non-empty prompt and a non-empty response")
    _, logits = model(ids[:, :-1])
    response_logits = logits[:, prompt_len - 1:, :]
    target = ids[:, prompt_len:]
    if response_logits.shape[:2] != target.shape:
        raise AssertionError("Off-by-one error in response-token alignment")
    return F.log_softmax(response_logits, dim=-1).gather(
        -1, target.unsqueeze(-1)
    ).squeeze(-1)


@torch.no_grad()
def response_ce(model, examples, stoi, device):
    """Cross entropy loss of a model on prompt-response pairs."""
    negative_logprob = 0.0
    n_tokens = 0
    for prompt, full, _ in examples:
        ids = torch.tensor([encode(full,stoi)], dtype=torch.long, device=device)
        logps = response_log_probs(model, ids, len(prompt))
        negative_logprob -= logps.sum().item()
        n_tokens += logps.numel()
    return negative_logprob / n_tokens


@torch.no_grad()
def sample_response(model, prompt, stoi, n_tokens, seed, device):
    torch.manual_seed(seed)
    if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    ids = torch.tensor([encode(prompt,stoi)], dtype=torch.long, device=device)
    for _ in range(n_tokens):
        _, logits = model(ids)
        next_id = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def evaluate_generation(model, reference, reward, examples, stoi, n_tokens, seed, device):
    """Estimate E[R(x,y)] and sampled per-token log(pi/pi_SFT) for y~model."""
    reward_sum = 0.0
    logratio_sum = 0.0
    total_tokens = 0
    for i, (prompt, _, _) in enumerate(examples):
        ids = sample_response(model, prompt, stoi, n_tokens, seed + i, device)
        scores = reward(ids, torch.ones_like(ids, dtype=torch.long))
        reward_sum += scores.item()
        new_logps = response_log_probs(model, ids, len(prompt))
        ref_logps = response_log_probs(reference, ids, len(prompt))
        logratio_sum += (new_logps - ref_logps).sum().item()
        total_tokens += new_logps.numel()
    return reward_sum / len(examples), logratio_sum / total_tokens


def ppo_dirs(experiments_dir):
    return {p for p in experiments_dir.glob("ppo_beta_*")
            if (p / "tinygpt_ppo.pt").is_file() and (p / "config.json").is_file()}


def run_ppo_sweep(args):
    if args.experiments_dir.resolve() != EXPERIMENTS.resolve():
        raise ValueError("Existing run_ppo.py always writes to ./experiments; "
                         "use the default --experiments-dir when --train-ppo is set")
    existing = ppo_dirs(args.experiments_dir)
    checkpoint_paths = []
    for beta in args.betas:
        command = [
            sys.executable, "-m", "scripts.run_ppo",
            "--sft-checkpoint", str(args.sft_checkpoint),
            "--reward-checkpoint", str(args.reward_checkpoint),
            "--train-data", str(args.train_data),
            "--val-data", str(args.val_data),
            "--beta", str(beta), "--seed", str(args.seed),
            "--num-epochs", str(args.ppo_epochs),
            "--ppo-updates", str(args.ppo_updates),
            "--max-new-tokens", str(args.max_new_tokens),
            "--max-train-examples", str(args.ppo_train_prompts),
            "--max-val-examples", str(args.ppo_val_prompts),
            "--device", args.device,
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        new_dirs = ppo_dirs(args.experiments_dir) - existing
        matches = [p for p in new_dirs if float(load_json(p / "config.json")["beta"]) == beta]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one new beta={beta} run, found {matches}")
        checkpoint_paths.append(matches[0] / "tinygpt_ppo.pt")
        existing.update(new_dirs)
    return checkpoint_paths


def find_existing(args):
    checkpoint_paths = []
    for beta in args.betas:
        matches = [p for p in ppo_dirs(args.experiments_dir)
                   if float(load_json(p / "config.json").get("beta", float("nan"))) == beta
                   and int(load_json(p / "config.json").get("seed", -1)) == args.seed]
        if not matches:
            raise FileNotFoundError(
                f"No PPO run for beta={beta}, seed={args.seed}; use --train-ppo"
            )
        matches.sort(key=lambda p: p.name)
        checkpoint_paths.append(matches[-1] / "tinygpt_ppo.pt")
    return checkpoint_paths


def verify_ppo_protocol(paths):
    """Do not silently compare PPO sweeps with different training settings."""
    fields = (
        "seed", "sft_checkpoint_path", "reward_checkpoint_path",
        "train_data_path", "val_data_path", "learning_rate", "num_epochs",
        "ppo_updates", "max_new_tokens", "epsilon", "gamma",
        "num_train_prompts", "num_val_prompts",
    )
    configs = [load_json(p.parent / "config.json") for p in paths]
    for cfg in configs[1:]:
        changed = [f for f in fields if cfg.get(f) != configs[0].get(f)]
        if changed:
            raise ValueError(f"PPO runs are not comparable; differing settings: {changed}")
    return configs


def bar_plot(rows, baseline_beta, output):
    import numpy as np
    selected = [r for r in rows if r["model"] == "SFT" or r["beta"] == baseline_beta]
    if len(selected) != 2:
        raise ValueError("Need one SFT and one PPO run at the chosen comparison beta")
    selected.sort(key=lambda r: 0 if r["model"] == "SFT" else 1)
    xs = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(xs - width / 2, [r["train_ce"] for r in selected], width, label="Train CE")
    ax.bar(xs + width / 2, [r["validation_ce"] for r in selected], width, label="Validation CE")
    ax.set_xticks(xs, ["SFT", f"PPO (beta={baseline_beta:g})"])
    ax.set_ylabel("Response-only cross entropy")
    ax.set_title("Experiment 1 · Same metric, same examples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def beta_plot(rows, metric, ylabel, title, output):
    rows = sorted((r for r in rows if r["model"] == "PPO"), key=lambda r: r["beta"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot([r["beta"] for r in rows], [r[metric] for r in rows], marker="o")
    ax.set_xlabel("KL coefficient beta")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Only two experiments: SFT vs PPO and KL beta sweep")
    parser.add_argument("--train-ppo", action="store_true", help="Train a fresh PPO run for every beta")
    parser.add_argument("--betas", default="0,0.05,0.2", help="Comma-separated values")
    parser.add_argument("--comparison-beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-examples", type=int, default=50)
    parser.add_argument("--val-examples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-updates", type=int, default=2)
    parser.add_argument("--ppo-train-prompts", type=int, default=30)
    parser.add_argument("--ppo-val-prompts", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS)
    parser.add_argument("--train-data", type=Path, default=DATA / "train_preferences.json")
    parser.add_argument("--val-data", type=Path, default=DATA / "validation_preferences.json")
    parser.add_argument("--sft-checkpoint", type=Path, default=Path("checkpoints/tinygpt_sft.pt"))
    parser.add_argument("--reward-checkpoint", type=Path, default=Path("checkpoints/reward_model.pt"))
    args = parser.parse_args()
    args.betas = [float(s) for s in args.betas.split(",")]
    if not args.betas or len(set(args.betas)) != len(args.betas) or any(b < 0 for b in args.betas):
        parser.error("--betas must be distinct nonnegative numbers")
    if args.comparison_beta not in args.betas:
        parser.error("--comparison-beta must be included in --betas")
    if args.max_new_tokens < 1 or args.train_examples < 1 or args.val_examples < 1:
        parser.error("--max-new-tokens and evaluation subset sizes must be positive")
    return args


def main():
    args = parse_args()
    device = torch.device(
        "mps" if (args.device == "auto" and torch.backends.mps.is_available())
        else "cpu" if args.device == "auto" else args.device
    )
    for p in (args.sft_checkpoint, args.reward_checkpoint, args.train_data, args.val_data):
        if not p.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {p}")
    paths = run_ppo_sweep(args) if args.train_ppo else find_existing(args)
    configs = verify_ppo_protocol(paths)
    if Path(configs[0]["sft_checkpoint_path"]).resolve() != args.sft_checkpoint.resolve():
        raise ValueError("PPO checkpoints were trained from a different SFT checkpoint path")
    if Path(configs[0]["reward_checkpoint_path"]).resolve() != args.reward_checkpoint.resolve():
        raise ValueError("PPO checkpoints used a different reward model path")

    checkpoint = torch.load(args.sft_checkpoint, map_location="cpu", weights_only=True)
    stoi = checkpoint["stoi"]
    reference, ref_config = load_llm(args.sft_checkpoint, stoi, device)
    reward, reward_context = load_reward(args.reward_checkpoint, stoi, device)
    context = min(ref_config["block_size"], reward_context)
    train_examples = choose_examples(args.train_data, stoi, context,
                                    args.max_new_tokens, args.train_examples, args.seed)
    # Exclude overlap with the ENTIRE original training split, not just the
    # small subset selected for train CE evaluation.
    train_keys = set()
    for ex in load_json(args.train_data):
        train_keys.add(("instruction", ex["instruction"]))
        if ex.get("post_id") is not None:
            train_keys.add(("post_id", str(ex["post_id"])))
    val_examples = choose_examples(args.val_data, stoi, context,
                                  args.max_new_tokens, args.val_examples,
                                  args.seed + 1000, train_keys=train_keys)
    print(f"Comparable examples: train={len(train_examples)}, val={len(val_examples)}")

    rows = []
    variants = [("SFT", None, args.sft_checkpoint)] + [
        ("PPO", beta, path) for beta, path in zip(args.betas, paths)
    ]
    for name, beta, path in variants:
        model = reference if name == "SFT" else load_llm(path, stoi, device)[0]
        if name == "PPO":
            # Individual PPO checkpoint settings must support every evaluation example.
            checkpoint_config = torch.load(path, map_location="cpu", weights_only=True)["config"]
            if any(len(full) > checkpoint_config["block_size"] for _, full, _ in train_examples + val_examples):
                raise ValueError(f"Evaluation length exceeds context for {path}")
        train_ce = response_ce(model, train_examples, stoi, device)
        val_ce = response_ce(model, val_examples, stoi, device)
        val_reward, val_sampled_kl = evaluate_generation(
            model, reference, reward, val_examples, stoi,
            args.max_new_tokens, args.seed + 2000, device
        )
        row = {
            "model": name, "beta": beta, "checkpoint": str(path),
            "train_ce": train_ce, "validation_ce": val_ce,
            "val_reward_score": val_reward, "val_sampled_kl_per_token": val_sampled_kl,
            "train_examples": len(train_examples), "val_examples": len(val_examples),
        }
        rows.append(row)
        print(f"{name} beta={beta}: trainCE={train_ce:.4f} "
              f"valCE={val_ce:.4f} reward={val_reward:.4f} sampledKL={val_sampled_kl:.4f}")
        if name != "SFT":
            del model

    out = args.experiments_dir / f"main_experiment_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True, exist_ok=False)
    save_csv(out / "results.csv", rows)
    bar_plot(rows, args.comparison_beta, out / "1_sft_vs_ppo_cross_entropy.png")
    beta_plot(rows, "val_reward_score", "Mean reward-model score", "Experiment 2 · Reward vs beta",
              out / "2_reward_vs_beta.png")
    beta_plot(rows, "val_sampled_kl_per_token", "Sampled log ratio to SFT (per token)",
              "Experiment 2 · Sampled KL estimate vs beta", out / "3_kl_vs_beta.png")
    with open(out / "notes.txt", "w", encoding="utf-8") as f:
        f.write(
            "Experiment 1 compares RESPONSE-ONLY teacher-forced cross entropy on the same "
            "accepted answers. Train/validation CE are measured AFTER training at final checkpoints; "
            "they are not epoch-wise PPO training losses. Lower CE means better likelihood on "
            "accepted responses, not necessarily better human preference.\n"
            "Experiment 2 uses the SAME learned reward model used in PPO optimization; score "
            "is a proxy, not independent quality. Sampled mean log(pi_beta/pi_SFT) on "
            "tokens drawn from pi_beta is an MC estimator of per-token KL; it may be "
            "negative with a small sample. Prompt sets, decoding length and seed are fixed.\n"
            "PPO clipped policy losses and SFT CE are NOT comparable as raw objectives.\n"
        )
    print("Saved 2 experiments to:", out)


if __name__ == "__main__":
    main()
