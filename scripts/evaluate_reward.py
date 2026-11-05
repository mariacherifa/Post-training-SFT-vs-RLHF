
######################################################################
#                     Evaluate Reward Model
######################################################################

import torch
import torch.nn.functional as F
import json
import csv
import argparse
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from src.models.reward_model import RewardModel
from src.data.utils import get_full_text


# -------------------------------------------------------
# 1. Load trained reward model
# -------------------------------------------------------

def load_reward_checkpoint(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    config = checkpoint["config"]
    stoi = checkpoint["stoi"]

    model = RewardModel(
        config["n_embd"],
        config["num_heads"],
        config["block_size"],
        config["vocab_size"],
        config["dropout"],
        config.get("num_layers", config.get("num_layer"))
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.requires_grad_(False)
    model.eval()

    return model, stoi, config["block_size"]


# -------------------------------------------------------
# 2. Evaluation dataset
# -------------------------------------------------------

class RewardEvaluationDataset(Dataset):

    def __init__(self, data_path, stoi, block_size):

        with open(data_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

        self.stoi = stoi
        self.examples = []

        for example in examples:

            chosen_text = get_full_text(
                {
                    "instruction": example["instruction"],
                    "response": example["accepted"]
                },
                task="SFT"
            )

            rejected_text = get_full_text(
                {
                    "instruction": example["instruction"],
                    "response": example["rejected"]
                },
                task="SFT"
            )

            vocab_ok = all(
                char in stoi
                for char in chosen_text + rejected_text
            )

            context_ok = (
                1 <= len(chosen_text) <= block_size
                and 1 <= len(rejected_text) <= block_size
            )

            if vocab_ok and context_ok:

                self.examples.append(
                    (chosen_text, rejected_text)
                )

        print("Total examples:", len(examples))
        print("Compatible examples:", len(self.examples))
        print("Filtered examples:", len(examples) - len(self.examples))

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, idx):

        chosen_text, rejected_text = self.examples[idx]

        chosen_ids = torch.tensor(
            [self.stoi[ch] for ch in chosen_text],
            dtype=torch.long
        )

        rejected_ids = torch.tensor(
            [self.stoi[ch] for ch in rejected_text],
            dtype=torch.long
        )

        return chosen_ids, rejected_ids


# -------------------------------------------------------
# 3. Collate evaluation batches
# -------------------------------------------------------

def reward_collate(batch, pad_id):

    chosen = [item[0] for item in batch]
    rejected = [item[1] for item in batch]

    chosen_mask = [
        torch.ones_like(ids)
        for ids in chosen
    ]

    rejected_mask = [
        torch.ones_like(ids)
        for ids in rejected
    ]

    return (
        pad_sequence(
            chosen,
            batch_first=True,
            padding_value=pad_id
        ),
        pad_sequence(
            rejected,
            batch_first=True,
            padding_value=pad_id
        ),
        pad_sequence(
            chosen_mask,
            batch_first=True,
            padding_value=0
        ),
        pad_sequence(
            rejected_mask,
            batch_first=True,
            padding_value=0
        )
    )


# -------------------------------------------------------
# 4. Evaluate preference pairs
# -------------------------------------------------------

@torch.no_grad()
def evaluate_preferences(model, data_loader, device):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    chosen_scores = []
    rejected_scores = []
    margins = []

    for (
        chosen_ids,
        rejected_ids,
        chosen_mask,
        rejected_mask
    ) in data_loader:

        chosen_ids = chosen_ids.to(device)
        rejected_ids = rejected_ids.to(device)

        chosen_mask = chosen_mask.to(device)
        rejected_mask = rejected_mask.to(device)

        # Compute reward scores
        chosen_rewards = model(
            chosen_ids,
            chosen_mask
        ).reshape(-1)

        rejected_rewards = model(
            rejected_ids,
            rejected_mask
        ).reshape(-1)

        # Reward margin
        margin = chosen_rewards - rejected_rewards

        # Bradley-Terry pairwise loss
        losses = F.softplus(-margin)

        total_loss += losses.sum().item()

        total_correct += (
            margin > 0
        ).sum().item()

        total_samples += margin.numel()

        chosen_scores.extend(
            chosen_rewards.cpu().tolist()
        )

        rejected_scores.extend(
            rejected_rewards.cpu().tolist()
        )

        margins.extend(
            margin.cpu().tolist()
        )

    if total_samples == 0:
        raise ValueError("No examples available for evaluation")

    metrics = {
        "num_examples": total_samples,
        "preference_accuracy": total_correct / total_samples,
        "pairwise_loss": total_loss / total_samples,
        "mean_margin": sum(margins) / total_samples,
        "mean_chosen_reward": sum(chosen_scores) / total_samples,
        "mean_rejected_reward": sum(rejected_scores) / total_samples
    }

    predictions = {
        "chosen_scores": chosen_scores,
        "rejected_scores": rejected_scores,
        "margins": margins
    }

    return metrics, predictions


# -------------------------------------------------------
# 5. Save evaluation results
# -------------------------------------------------------

def save_results(metrics, predictions, output_dir):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save summary metrics
    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # Save scores for individual examples
    with open(
        output_dir / "predictions.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "example_id",
            "chosen_reward",
            "rejected_reward",
            "margin",
            "correct"
        ])

        for i, (chosen, rejected, margin) in enumerate(
            zip(
                predictions["chosen_scores"],
                predictions["rejected_scores"],
                predictions["margins"]
            )
        ):

            writer.writerow([
                i,
                chosen,
                rejected,
                margin,
                int(margin > 0)
            ])

    # Plot reward distributions
    plt.figure(figsize=(9, 5))

    plt.hist(
        predictions["chosen_scores"],
        bins=25,
        alpha=0.6,
        label="Chosen"
    )

    plt.hist(
        predictions["rejected_scores"],
        bins=25,
        alpha=0.6,
        label="Rejected"
    )

    plt.xlabel("Predicted reward")
    plt.ylabel("Number of examples")
    plt.legend()

    plt.savefig(
        output_dir / "reward_distributions.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    # Plot reward margins
    plt.figure(figsize=(9, 5))

    plt.hist(
        predictions["margins"],
        bins=30
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Decision boundary"
    )

    plt.xlabel("Chosen reward - Rejected reward")
    plt.ylabel("Number of examples")
    plt.legend()

    plt.savefig(
        output_dir / "reward_margins.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()


# -------------------------------------------------------
# 6. Main evaluation
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation"
    )

    args = parser.parse_args()

    # Paths
    reward_checkpoint_path = Path(
        "checkpoints/reward_model.pt"
    )

    sft_checkpoint_path = Path(
        "checkpoints/tinygpt_sft.pt"
    )

    data_path = Path(
        f"data/shp_final/{args.split}_reward.json"
    )

    output_dir = Path(
        f"experiments/reward_evaluation/{args.split}"
    )

    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Load trained reward model
    model, stoi, block_size = load_reward_checkpoint(
        reward_checkpoint_path,
        device
    )

    # Verify vocabulary matches SFT
    sft_checkpoint = torch.load(
        sft_checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if sft_checkpoint["stoi"] != stoi:
        raise ValueError(
            "Reward and SFT vocabularies do not match"
        )

    # Prepare evaluation dataset
    dataset = RewardEvaluationDataset(
        data_path,
        stoi,
        block_size
    )

    pad_id = stoi.get(" ", 0)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: reward_collate(
            batch,
            pad_id
        )
    )

    # Evaluate
    metrics, predictions = evaluate_preferences(
        model,
        data_loader,
        device
    )

    # Save results
    save_results(
        metrics,
        predictions,
        output_dir
    )

    # Print metrics
    print("\nReward Model Evaluation")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    main()
