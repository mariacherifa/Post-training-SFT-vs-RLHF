##########################################################################
#                 Run a controlled SFT vs PPO experiment suite
##########################################################################

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def csv_values(text, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def run_command(arguments):
    print("\n>>", " ".join(str(a) for a in arguments), flush=True)
    subprocess.run([str(a) for a in arguments], check=True)


def main():
    parser = argparse.ArgumentParser(description="Train optional PPO sweeps, evaluate models, aggregate")
    parser.add_argument("--train", action="store_true",
                        help="Run PPO training before evaluation. Without this, reuse saved PPO runs")
    parser.add_argument("--betas", default="0.01,0.05,0.1")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-updates", type=int, default=2)
    parser.add_argument("--train-examples", type=int, default=10,
                        help="0 uses all compatible SHP PPO training prompts")
    parser.add_argument("--val-examples", type=int, default=4,
                        help="0 uses all compatible SHP validation prompts")
    parser.add_argument("--eval-prompts", type=int, default=12,
                        help="0 uses all compatible prompts")
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--eval-max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--pretrained-checkpoint", type=Path,
                        default=Path("checkpoints/tinygpt_2048_pretrained.pt"))
    parser.add_argument("--sft-checkpoint", type=Path,
                        default=Path("checkpoints/tinygpt_sft.pt"))
    parser.add_argument("--reward-checkpoint", type=Path,
                        default=Path("checkpoints/reward_model.pt"))
    args = parser.parse_args()

    betas = csv_values(args.betas, float)
    seeds = csv_values(args.seeds, int)
    if not betas or not seeds:
        parser.error("Provide at least one beta and seed")
    if any(beta < 0 for beta in betas):
        parser.error("beta cannot be negative")
    if args.eval_prompts < 0 or args.eval_samples < 1:
        parser.error("--eval-prompts >= 0, --eval-samples >= 1")

    # Keep separate outputs per suite; do not overwrite completed runs.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    eval_dir = args.experiments_dir / "model_evaluation" / f"{args.split}_suite_{timestamp}"
    aggregate_dir = args.experiments_dir / "final_comparison" / f"suite_{timestamp}"

    existing = {p.resolve() for p in args.experiments_dir.glob("ppo_beta_*/tinygpt_ppo.pt")}
    if args.train:
        for beta in betas:
            for seed in seeds:
                run_command([
                    sys.executable, "-m", "scripts.run_ppo",
                    "--beta", beta,
                    "--seed", seed,
                    "--num-epochs", args.ppo_epochs,
                    "--ppo-updates", args.ppo_updates,
                    "--max-new-tokens", args.max_new_tokens,
                    "--max-train-examples", args.train_examples,
                    "--max-val-examples", args.val_examples,
                    "--device", args.device,
                    "--sft-checkpoint", args.sft_checkpoint,
                    "--reward-checkpoint", args.reward_checkpoint,
                ])
    else:
        print("Reusing completed ppo_beta_* runs in:", args.experiments_dir)

    newly_created = sorted(
        (p for p in args.experiments_dir.glob("ppo_beta_*/tinygpt_ppo.pt")
         if p.resolve() not in existing),
        key=lambda p: str(p)
    )
    eval_command = [
        sys.executable, "-m", "scripts.evaluate_models",
        "--pretrained-checkpoint", args.pretrained_checkpoint,
        "--sft-checkpoint", args.sft_checkpoint,
        "--reward-checkpoint", args.reward_checkpoint,
        "--experiments-dir", args.experiments_dir,
        "--split", args.split,
        "--num-prompts", args.eval_prompts,
        "--max-new-tokens", args.max_new_tokens,
        "--max-prompt-tokens", args.eval_max_prompt_tokens,
        "--samples-per-prompt", args.eval_samples,
        "--device", args.device,
        "--output-dir", eval_dir,
    ]
    # When training a sweep, do not mix these results with old debug runs.
    if args.train:
        if not newly_created:
            raise RuntimeError("PPO training produced no new ppo_beta_* checkpoints")
        eval_command.append("--only-specified-ppo")
        for checkpoint_path in newly_created:
            eval_command.extend(["--ppo-checkpoint", checkpoint_path])
    run_command(eval_command)

    run_command([
        sys.executable, "-m", "scripts.aggregate_experiments",
        "--experiments-dir", args.experiments_dir,
        "--evaluation-dir", eval_dir,
        "--output-dir", aggregate_dir,
    ])
    print("\nEvaluation results:", eval_dir)
    print("Aggregate figures and tables:", aggregate_dir)


if __name__ == "__main__":
    main()
