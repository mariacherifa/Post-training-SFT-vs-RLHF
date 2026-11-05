# Two experiments: (1) SFT vs PPO cross-entropy,
# (2) KL regularization: raw reward, signed KL contribution, objective, sampled KL.
# Plot 2 is a decomposition of the RLHF evaluation objective, NOT PPO's
# clipped policy loss + 0.5 * value loss.


import argparse
import csv
import json
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
from scripts.run_ppo import load_reward_model

# ------------------------- Settings -------------------------
BETAS = [0.0, 1.0, 2.5]
COMPARISON_BETA = 1.0
SEED = 42
PPO_EPOCHS = 5
PPO_UPDATES = 2
PPO_TRAIN_PROMPTS = 200	
PPO_VAL_PROMPTS = 50
EVAL_TRAIN_EXAMPLES = 200
EVAL_VAL_EXAMPLES = 200
MAX_NEW_TOKENS = 16

EXPERIMENTS = Path("experiments")
SFT_CHECKPOINT = Path("checkpoints/tinygpt_sft.pt")
REWARD_CHECKPOINT = Path("checkpoints/reward_model.pt")
TRAIN_DATA = Path("data/shp_processed/train_preferences.json")
VAL_DATA = Path("data/shp_processed/validation_preferences.json")


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_model(checkpoint_path, device, vocabulary=None):
    """Load either SFT or PPO using your existing TinyGPT checkpoint format."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if vocabulary is not None and checkpoint["stoi"] != vocabulary:
        raise ValueError(f"Vocabulary mismatch in {checkpoint_path}")
    config = checkpoint["config"]
    model = TinyGPT(
        n_embd=config["n_embd"], num_heads=config["num_heads"],
        block_size=config["block_size"], vocab_size=config["vocab_size"],
        dropout=config["dropout"],
        num_layers=config.get("num_layers", config.get("num_layer")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().requires_grad_(False)
    return model, checkpoint["stoi"], config["block_size"]


def select_examples(data_path, vocabulary, context_length, count, seed,
                    excluded_instructions=set(), excluded_posts=set()):
    """One fixed set of eligible (prompt, accepted answer) pairs for all models."""
    data = read_json(data_path)
    random.Random(seed).shuffle(data)
    examples, seen_instructions, seen_posts = [], set(excluded_instructions), set(excluded_posts)
    for item in data:
        instruction = item["instruction"]
        post = str(item["post_id"]) if item.get("post_id") is not None else None
        if instruction in seen_instructions or (post is not None and post in seen_posts):
            continue
        prompt = get_full_text({"instruction": instruction, "response": ""}, task="SFT")
        full = get_full_text({"instruction": instruction, "response": item["accepted"]}, task="SFT")
        if not full.startswith(prompt):
            raise ValueError("The formatted prompt is not a prefix of the full example")
        if not (1 <= len(prompt) < len(full) <= context_length):
            continue
        if len(prompt) + MAX_NEW_TOKENS > context_length:
            continue
        if any(character not in vocabulary for character in full):
            continue
        examples.append((prompt, full))
        seen_instructions.add(instruction)
        if post is not None:
            seen_posts.add(post)
        if len(examples) >= count:
            break
    if not examples:
        raise ValueError(f"No compatible examples in {data_path}")
    return examples


def encode(text, vocabulary, device):
    return torch.tensor([[vocabulary[character] for character in text]],
                        dtype=torch.long, device=device)


@torch.no_grad()
def response_log_probs(model, full_ids, prompt_length):
    """Log probabilities of response tokens (teacher-forced for references)."""
    _, logits = model(full_ids[:, :-1])
    targets = full_ids[:, prompt_length:]
    return F.log_softmax(logits[:, prompt_length - 1:, :], dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)


@torch.no_grad()
def cross_entropy(model, examples, vocabulary, device):
    total_loss, total_tokens = 0.0, 0
    for prompt, full in examples:
        log_probs = response_log_probs(model, encode(full, vocabulary, device), len(prompt))
        total_loss -= log_probs.sum().item()
        total_tokens += log_probs.numel()
    return total_loss / total_tokens


@torch.no_grad()
def evaluate_generation(model, reference, reward_model, examples, vocabulary, device):
    """Sample responses; measure mean reward and sampled KL-to-SFT per token."""
    total_reward, total_log_ratio, total_tokens = 0.0, 0.0, 0
    for index, (prompt, _) in enumerate(examples):
        torch.manual_seed(SEED + 2000 + index)
        if device.type == "mps":
            torch.mps.manual_seed(SEED + 2000 + index)
        full_ids = encode(prompt, vocabulary, device)
        for _ in range(MAX_NEW_TOKENS):
            _, logits = model(full_ids)
            next_token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            full_ids = torch.cat([full_ids, next_token], dim=1)
        total_reward += reward_model(full_ids, torch.ones_like(full_ids)).item()
        policy_log_probs = response_log_probs(model, full_ids, len(prompt))
        reference_log_probs = response_log_probs(reference, full_ids, len(prompt))
        total_log_ratio += (policy_log_probs - reference_log_probs).sum().item()
        total_tokens += policy_log_probs.numel()
        
    return total_reward / len(examples), total_log_ratio / total_tokens


def get_ppo_checkpoints(train_ppo):
    """Train with run_ppo.py, or find existing runs for each beta."""
    checkpoints = {}
    for beta in BETAS:
        existing = set(EXPERIMENTS.glob("ppo_beta_*/tinygpt_ppo.pt"))
        if train_ppo:
            subprocess.run([
                sys.executable, "-m", "scripts.run_ppo",
                "--beta", str(beta), "--seed", str(SEED),
                "--num-epochs", str(PPO_EPOCHS),
                "--ppo-updates", str(PPO_UPDATES),
                "--max-new-tokens", str(MAX_NEW_TOKENS),
                "--max-train-examples", str(PPO_TRAIN_PROMPTS),
                "--max-val-examples", str(PPO_VAL_PROMPTS),
            ], check=True)
        candidates = set(EXPERIMENTS.glob("ppo_beta_*/tinygpt_ppo.pt"))
        if train_ppo:
            candidates -= existing
        matches = []
        for checkpoint_path in candidates:
            config_path = checkpoint_path.parent / "config.json"
            if config_path.is_file():
                config = read_json(config_path)
                if config.get("beta") == beta and config.get("seed") == SEED:
                    matches.append(checkpoint_path)
        if not matches:
            raise FileNotFoundError(f"Missing PPO checkpoint for beta={beta}. Try --train-ppo")
        checkpoints[beta] = sorted(matches)[-1]

    # Don't accidentally compare runs trained under different settings.
    settings = ("sft_checkpoint_path", "reward_checkpoint_path", "train_data_path",
                "val_data_path", "seed", "learning_rate", "num_epochs", "ppo_updates",
                "max_new_tokens", "epsilon", "gamma", "num_train_prompts", "num_val_prompts")
    configs = [read_json(path.parent / "config.json") for path in checkpoints.values()]
    if any(any(config.get(key) != configs[0].get(key) for key in settings) for config in configs[1:]):
        raise ValueError("PPO runs do not use the same settings. Run with --train-ppo.")
    if Path(configs[0]["sft_checkpoint_path"]).resolve() != SFT_CHECKPOINT.resolve():
        raise ValueError("PPO was trained using another SFT checkpoint")
    if Path(configs[0]["reward_checkpoint_path"]).resolve() != REWARD_CHECKPOINT.resolve():
        raise ValueError("PPO was trained using another reward checkpoint")
    return checkpoints


def save_results(rows, output_dir):
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    sft = rows[0]
    comparison = next(row for row in rows if row["beta"] == COMPARISON_BETA)
    plt.figure(figsize=(7, 4))
    plt.bar([-0.18, 0.82], [sft["train_ce"], comparison["train_ce"]], 0.36, label="Train CE")
    plt.bar([0.18, 1.18], [sft["validation_ce"], comparison["validation_ce"]],
            0.36, label="Validation CE")
    plt.xticks([0, 1], ["SFT", f"PPO (beta={COMPARISON_BETA})"])
    plt.ylabel("Response-only cross-entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "1_sft_vs_ppo.png")
    plt.close()

    ppo_rows = rows[1:]
    betas = [row["beta"] for row in ppo_rows]

    # Experiment 2, plot A: break down the regularized RLHF objective.
    # J_beta = mean_reward - beta * mean(sum of sampled log policy/ref ratios).
    # The negative KL contribution is plotted with its sign, so that
    # regularized_reward = raw_reward + negative_kl_contribution.
    plt.figure(figsize=(7, 4))
    plt.plot(betas, [row["validation_reward"] for row in ppo_rows],
             marker="o", label="Raw reward")
    plt.plot(betas, [-row["kl_penalty"] for row in ppo_rows],
             marker="o", label="KL contribution (-beta × sampled KL)")
    plt.plot(betas, [row["regularized_reward"] for row in ppo_rows],
             marker="o", label="Regularized objective")
    plt.xlabel("KL regularization coefficient beta")
    plt.ylabel("Mean contribution per generated response")
    plt.title("Reward and KL contribution to the RLHF objective")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "2_objective_vs_beta.png")
    plt.close()

    # Experiment 2, plot B: keep the original sampled KL diagnostic.
    plt.figure(figsize=(7, 4))
    plt.plot(betas, [row["sampled_kl"] for row in ppo_rows], marker="o")
    plt.xlabel("KL regularization coefficient beta")
    plt.ylabel("Sampled KL estimate per token")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "3_kl_vs_beta.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ppo", action="store_true")
    args = parser.parse_args()
    if COMPARISON_BETA not in BETAS:
        raise ValueError("COMPARISON_BETA must belong to BETAS")
    for path in (SFT_CHECKPOINT, REWARD_CHECKPOINT, TRAIN_DATA, VAL_DATA):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ppo_checkpoints = get_ppo_checkpoints(args.train_ppo)
    sft_model, vocabulary, sft_context = load_model(SFT_CHECKPOINT, device)
    checkpoint = torch.load(SFT_CHECKPOINT, map_location="cpu", weights_only=True)
    reward_model, reward_config = load_reward_model(
        REWARD_CHECKPOINT, checkpoint["config"], vocabulary, device
    )
    context_length = min(sft_context, reward_config["block_size"])

    train_examples = select_examples(TRAIN_DATA, vocabulary, context_length,
                                     EVAL_TRAIN_EXAMPLES, SEED)
    original_training_data = read_json(TRAIN_DATA)
    val_examples = select_examples(
        VAL_DATA, vocabulary, context_length, EVAL_VAL_EXAMPLES, SEED + 1000,
        {item["instruction"] for item in original_training_data},
        {str(item["post_id"]) for item in original_training_data if item.get("post_id") is not None},
    )

    rows = []
    for name, beta, checkpoint_path in [("SFT", None, SFT_CHECKPOINT)] + [
        ("PPO", beta, ppo_checkpoints[beta]) for beta in BETAS
    ]:
        model = sft_model if name == "SFT" else load_model(checkpoint_path, device, vocabulary)[0]
        train_ce = cross_entropy(model, train_examples, vocabulary, device)
        validation_ce = cross_entropy(model, val_examples, vocabulary, device)
        validation_reward, sampled_kl = evaluate_generation(
            model, sft_model, reward_model, val_examples, vocabulary, device
        )
        # The KL estimate above is per token. All generated responses have
        # exactly MAX_NEW_TOKENS tokens, so multiply by that length to get
        # the estimated KL-related cost per complete response.
        # For the SFT baseline (beta=None), the regularization cost is zero.
        kl_penalty = 0.0 if beta is None else beta * MAX_NEW_TOKENS * sampled_kl
        regularized_reward = validation_reward - kl_penalty

        rows.append({"model": name, "beta": beta, "checkpoint": str(checkpoint_path),
                     "train_ce": train_ce, "validation_ce": validation_ce,
                     "validation_reward": validation_reward, "sampled_kl": sampled_kl,
                     "kl_penalty": kl_penalty,
                     "regularized_reward": regularized_reward})
        print(f"{name} beta={beta}: trainCE={train_ce:.4f}, valCE={validation_ce:.4f}, "
              f"reward={validation_reward:.4f}, sampledKL={sampled_kl:.4f}, "
              f"KLcost={kl_penalty:.4f}, objective={regularized_reward:.4f}")

    output_dir = EXPERIMENTS / f"main_experiment_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(rows, output_dir)
    print("Saved results and 3 plots in:", output_dir)


if __name__ == "__main__":
    main()
