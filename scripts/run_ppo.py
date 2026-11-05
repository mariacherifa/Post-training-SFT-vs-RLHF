##########################################################################
#                       PPO Experiment
#                   Load existing Reward Model
##########################################################################

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.utils import get_full_text
from src.models.reference_policies_rlhf import reference_policy
from src.models.reward_model import RewardModel
from src.models.value_model import ValueModel
from src.trainers.ppo_training import PPOTrain


# -------------------------------------------------------
# 1. Format prompts exactly as in SFT
# -------------------------------------------------------

def format_prompt(instruction):
    return get_full_text(
        {"instruction": instruction, "response": ""},
        task="SFT"
    )


# -------------------------------------------------------
# 2. Select compatible prompts (deterministic ordering)
# -------------------------------------------------------

def prepare_prompts(data_path, stoi, block_size, max_new_tokens, max_examples=None):
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    prompts = []
    seen = set()

    for example in examples:
        prompt = format_prompt(example["instruction"])

        vocab_ok = all(char in stoi for char in prompt)
        context_ok = (1 <= len(prompt) and
                      len(prompt) + max_new_tokens <= block_size)

        if vocab_ok and context_ok and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)

        if max_examples is not None and len(prompts) >= max_examples:
            break

    if not prompts:
        raise ValueError(f"No compatible PPO prompts in {data_path}")

    print(f"{data_path}: {len(prompts)} compatible unique prompts "
          f"(out of {len(examples)} input examples)")
    return prompts


class PPOPromptDataset(Dataset):

    def __init__(self, prompts, stoi):
        self.prompts = prompts
        self.stoi = stoi

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return torch.tensor(
            [self.stoi[char] for char in self.prompts[idx]],
            dtype=torch.long
        )


# -------------------------------------------------------
# 3. Load an already-trained reward model
# -------------------------------------------------------

def load_reward_model(checkpoint_path, policy_config, policy_stoi, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    # A matching vocabulary size is insufficient; token IDs must also match.
    if "stoi" not in checkpoint:
        raise ValueError(
            "Reward checkpoint does not contain 'stoi'. Save the original "
            "token-to-ID mapping in the reward checkpoint before using it for PPO."
        )
    if checkpoint["stoi"] != policy_stoi:
        raise ValueError("Reward checkpoint and SFT have different stoi mappings")

    reward_config = checkpoint.get("config", {})

    # Support both flat checkpoint fields and nested config, including
    # num_layer (reward model) versus num_layers (TinyGPT config).
    def get_arch_param(name, alternate=None):
        for source in (reward_config, checkpoint, policy_config):
            for key in (name, alternate):
                if key is not None and key in source:
                    return source[key]
        raise KeyError(f"Reward checkpoint is missing architecture parameter {name}")

    arch = {
        "n_embd": get_arch_param("n_embd"),
        "num_heads": get_arch_param("num_heads"),
        "block_size": get_arch_param("block_size"),
        "vocab_size": get_arch_param("vocab_size"),
        "dropout": get_arch_param("dropout"),
        "num_layers": get_arch_param("num_layer", "num_layers"),
    }

    if arch["vocab_size"] != len(policy_stoi):
        raise ValueError("Reward model vocabulary size differs from SFT")

    reward_model = RewardModel(
        arch["n_embd"],
        arch["num_heads"],
        arch["block_size"],
        arch["vocab_size"],
        arch["dropout"],
        arch["num_layers"]
    )

    
    reward_model.load_state_dict(checkpoint["model_state_dict"])
    reward_model = reward_model.to(device)
    reward_model.requires_grad_(False)
    reward_model.eval()

    return reward_model, arch


# -------------------------------------------------------
# 4. Save a checkpoint without corrupting an existing one
# -------------------------------------------------------

def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary_path)

    # Verify readability before replacing the destination.
    torch.load(temporary_path, map_location="cpu", weights_only=True)
    os.replace(temporary_path, path)


# -------------------------------------------------------
# 5. Save all PPO metrics in one CSV
# -------------------------------------------------------

def save_metrics_csv(history, output_path):
    output_path = Path(output_path)

    num_epochs = len(history["train_loss"])
    eval_epochs = history.get("eval_epochs", [])

    train_keys = [
        "train_loss", "policy_loss", "value_loss", "mean_reward",
        "mean_kl", "clip_fraction", "response_length"
    ]
    val_keys = [
        "val_loss", "val_policy_loss", "val_value_loss",
        "val_mean_reward", "val_mean_kl", "val_response_length"
    ]

    # Older PPOTrain histories may omit some diagnostic fields.
    train_keys = [key for key in train_keys if key in history]
    val_keys = [key for key in val_keys if key in history]

    for key in train_keys:
        if len(history[key]) != num_epochs:
            raise ValueError(f"{key} has {len(history[key])} values; "
                             f"expected {num_epochs}")
    for key in val_keys:
        if len(history[key]) != len(eval_epochs):
            raise ValueError(f"{key} length must equal eval_epochs length")

    eval_position = {epoch: idx for idx, epoch in enumerate(eval_epochs)}

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + train_keys + val_keys)

        for epoch in range(1, num_epochs + 1):
            row = [epoch]
            row += [history[key][epoch - 1] for key in train_keys]
            idx = eval_position.get(epoch)
            row += [history[key][idx] if idx is not None else ""
                    for key in val_keys]
            writer.writerow(row)


# -------------------------------------------------------
# 6. Save PPO learning curves
# -------------------------------------------------------

def save_learning_curves(history, output_dir):
    output_dir = Path(output_dir)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    eval_epochs = history.get("eval_epochs", [])

    plots = [
        ("total_loss.png", "Combined PPO and value loss",
         "train_loss", "val_loss"),
        ("policy_loss.png", "PPO policy loss",
         "policy_loss", "val_policy_loss"),
        ("value_loss.png", "Value loss",
         "value_loss", "val_value_loss"),
        ("reward.png", "Reward-model score per response",
         "mean_reward", "val_mean_reward"),
        ("kl.png", "Sampled log-ratio (KL estimate) per token",
         "mean_kl", "val_mean_kl"),
        ("response_length.png", "Generated response length (tokens)",
         "response_length", "val_response_length"),
        ("clip_fraction.png", "PPO clip fraction",
         "clip_fraction", None),
    ]

    for filename, ylabel, train_key, val_key in plots:
        if train_key not in history:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, history[train_key], marker="o", label="Train")

        if val_key and val_key in history and len(history[val_key]) == len(eval_epochs):
            ax.plot(eval_epochs, history[val_key], marker="o", label="Validation")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)


# -------------------------------------------------------
# 7. Parse experiment arguments
# -------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO experiment using fixed SFT and reward checkpoints"
    )

    parser.add_argument("--sft-checkpoint", type=Path,
                        default=Path("checkpoints/tinygpt_sft.pt"))
    parser.add_argument("--reward-checkpoint", type=Path,
                        default=Path("checkpoints/reward_model.pt"))
    parser.add_argument("--train-data", type=Path,
                        default=Path("data/shp_processed/train_preferences.json"))
    parser.add_argument("--val-data", type=Path,
                        default=Path("data/shp_processed/validation_preferences.json"))

    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--ppo-updates", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)

    # 0 means use all compatible prompts; defaults are small debug subsets.
    parser.add_argument("--max-train-examples", type=int, default=10)
    parser.add_argument("--max-val-examples", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"],
                        default="auto")

    args = parser.parse_args()

    if args.beta < 0:
        parser.error("--beta must be non-negative")
    if args.num_epochs < 1 or args.eval_every < 1 or args.ppo_updates < 1:
        parser.error("--num-epochs, --eval-every, and --ppo-updates must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.epsilon <= 0 or args.epsilon >= 1:
        parser.error("--epsilon must be strictly between 0 and 1")
    if not (0 <= args.gamma <= 1):
        parser.error("--gamma must be between 0 and 1")
    if args.max_train_examples < 0 or args.max_val_examples < 0:
        parser.error("--max-*-examples must be >= 0 (0 means all)")

    return args


# -------------------------------------------------------
# 8. Main PPO experiment
# -------------------------------------------------------

def main():
    args = parse_args()

    # -------------------------------------------------------
    # Seed and device
    # -------------------------------------------------------

    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available")

    

    # -------------------------------------------------------
    #  Paths and unique experiment directory
    # -------------------------------------------------------

    for path in (
        args.sft_checkpoint,
        args.reward_checkpoint,
        args.train_data,
        args.val_data,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_name = (
        f"ppo_beta_{args.beta:g}_seed_{args.seed}_{timestamp}"
    )
    output_dir = Path("experiments") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=False)

    # -------------------------------------------------------
    #  Load SFT model and vocabulary
    # -------------------------------------------------------

    sft_checkpoint = torch.load(
        args.sft_checkpoint, map_location="cpu", weights_only=True
    )
    config = sft_checkpoint["config"]
    stoi = sft_checkpoint["stoi"]
    itos = sft_checkpoint["itos"]

    if config["vocab_size"] != len(stoi):
        raise ValueError("SFT config vocab_size does not match stoi")

    policy, ref_policy = reference_policy(
        config, args.sft_checkpoint, device
    )
    policy = policy.to(device)
    ref_policy = ref_policy.to(device)
    ref_policy.requires_grad_(False)
    ref_policy.eval()

    # Every run starts from the SAME SFT policy (never a previous PPO run).
    for p, ref_p in zip(policy.parameters(), ref_policy.parameters()):
        if not torch.equal(p.detach().cpu(), ref_p.detach().cpu()):
            raise ValueError("PPO policy and reference do not start from equal weights")

    # -------------------------------------------------------
    #  Load the already-trained reward model
    # -------------------------------------------------------

    reward_model, reward_arch = load_reward_model(
        args.reward_checkpoint, config, stoi, device
    )
    effective_block_size = min(
        config["block_size"], reward_arch["block_size"]
    )

    if args.max_new_tokens >= effective_block_size:
        raise ValueError("max_new_tokens must be smaller than context size")

    # -------------------------------------------------------
    #  Initialize trainable value model
    # -------------------------------------------------------

    value_model = ValueModel(
        config["n_embd"],
        config["num_heads"],
        config["block_size"],
        config["vocab_size"],
        config["dropout"],
        config["num_layers"]
    ).to(device)

    # -------------------------------------------------------
    #  Prepare fixed prompt selections for this run
    # -------------------------------------------------------

    max_train = args.max_train_examples or None
    max_val = args.max_val_examples or None

    prompts_train = prepare_prompts(
        args.train_data, stoi, effective_block_size,
        args.max_new_tokens, max_examples=max_train
    )
    prompts_val = prepare_prompts(
        args.val_data, stoi, effective_block_size,
        args.max_new_tokens, max_examples=max_val
    )

    # Avoid overlapping prompts, even if SHP contains repeated instructions.
    overlap = set(prompts_train) & set(prompts_val)
    if overlap:
        raise ValueError(
            f"Train/validation PPO prompt leakage: {len(overlap)} duplicates. "
            "Deduplicate your SHP splits before running experiments."
        )

    for filename, prompts in (
        ("train_prompts.json", prompts_train),
        ("val_prompts.json", prompts_val),
    ):
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

    # Our PPOTrain uses uniform-length tensors; batch_size=1 avoids padding.
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PPOPromptDataset(prompts_train, stoi),
        batch_size=1, shuffle=True, generator=shuffle_generator
    )
    val_loader = DataLoader(
        PPOPromptDataset(prompts_val, stoi),
        batch_size=1, shuffle=False
    )

    # -------------------------------------------------------
    #  PPO trainers and optimizers
    # -------------------------------------------------------

    policy_optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate
    )
    value_optimizer = torch.optim.AdamW(
        value_model.parameters(), lr=args.learning_rate
    )

    trainer = PPOTrain(
        policy,
        ref_policy,
        reward_model,
        value_model,
        policy_optimizer,
        value_optimizer,
        effective_block_size,
        args.max_new_tokens,
        args.beta,
        args.epsilon,
        args.gamma,
        device,
        ppo_updates=args.ppo_updates
    )

    # Save configuration BEFORE training, so interrupted runs have metadata.
    experiment_config = {
        "sft_checkpoint_path": str(args.sft_checkpoint.resolve()),
        "reward_checkpoint_path": str(args.reward_checkpoint.resolve()),
        "train_data_path": str(args.train_data.resolve()),
        "val_data_path": str(args.val_data.resolve()),
        "seed": args.seed,
        "device": str(device),
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "eval_every": args.eval_every,
        "ppo_updates": args.ppo_updates,
        "max_new_tokens": args.max_new_tokens,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "batch_size": 1,
        "max_train_examples": args.max_train_examples,
        "max_val_examples": args.max_val_examples,
        "num_train_prompts": len(prompts_train),
        "num_val_prompts": len(prompts_val),
        "block_size": effective_block_size,
        "model_config": config,
        "reward_model_config": reward_arch,
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)

    # -------------------------------------------------------
    #  Train PPO
    # -------------------------------------------------------

    print("Starting PPO training:", experiment_name)
    print(f"Beta={args.beta:g}, seed={args.seed}")

    history = trainer.full_training(
        train_loader,
        val_loader,
        args.num_epochs,
        eval_every=args.eval_every
    )

    # -------------------------------------------------------
    #  Save checkpoint and the complete history
    # -------------------------------------------------------

    ppo_checkpoint = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": policy_optimizer.state_dict(),
        "value_model_state_dict": value_model.state_dict(),
        "value_optimizer_state_dict": value_optimizer.state_dict(),
        "config": config,
        "stoi": stoi,
        "itos": itos,
        "epoch": args.num_epochs,
        "reward_checkpoint_path": str(args.reward_checkpoint),
        "sft_checkpoint_path": str(args.sft_checkpoint),
        "max_new_tokens": args.max_new_tokens,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "ppo_updates": args.ppo_updates,
        "seed": args.seed,
    }

    checkpoint_path = output_dir / "tinygpt_ppo.pt"
    save_checkpoint(ppo_checkpoint, checkpoint_path)

    torch.save(history, output_dir / "history.pt")
    save_metrics_csv(history, output_dir / "metrics.csv")
    #save_learning_curves(history, output_dir)

    
    return history


if __name__ == "__main__":
    main()
