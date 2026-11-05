
######################################################################
#                       SFT Experiment
######################################################################

import torch
import json

from pathlib import Path
from torch.utils.data import DataLoader

from src.models.policy import load_policy
from src.data.SFT_dataset import SFTDataset
from src.trainers.training_sft import TrainSFT
from src.data.utils import get_full_text
from torch.utils.data import Subset
import csv
from datetime import datetime

import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt

def compatible_examples(checkpoint_path, data_path, output_path=None):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    stoi = checkpoint["stoi"]
    block_size = checkpoint["config"]["block_size"]

    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    examples_right_keys = [
        {
            "instruction": example["instruction"],
            "response": example["accepted"]
        }
        for example in examples
    ]

    compatible_examples = []

    for example in examples_right_keys:

        text = get_full_text(example, task="SFT")

        vocab_ok = all(char in stoi for char in text)
        context_ok = len(text) - 1 <= block_size

        if vocab_ok and context_ok:
            compatible_examples.append(example)

    # Save compatible examples
    output_dir = Path("data/shp_processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "train_sft_compatible.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(compatible_examples, f, indent=2, ensure_ascii=False)

    return output_path


def main():

    # -------------------------------------------------------
    # 1. Define paths
    # -------------------------------------------------------

    checkpoint_path = Path(
        "checkpoints/tinygpt_2048_pretrained.pt"
    )

    train_path = Path(
        "data/shp_processed/train_preferences.json"
    )

    val_path = Path(
        "data/shp_processed/validation_preferences.json"
    )

    output_path = Path(
        "checkpoints/tinygpt_sft.pt"
    )

    # -------------------------------------------------------
    # 2. Define hyperparameters
    # -------------------------------------------------------

    batch_size = 1
    learning_rate = 1e-4
    num_epochs = 2
    eval_every = 1
    device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

    # -------------------------------------------------------
    # 3. Load checkpoint and configuration
    # -------------------------------------------------------

    checkpoints = torch.load(checkpoint_path,
                             map_location=device, 
                             weights_only=True)

    stoi = checkpoints["stoi"]
    itos = checkpoints["itos"]
    config = checkpoints["config"]

    # -------------------------------------------------------
    # 4. Initialize TinyGPT policy
    # -------------------------------------------------------

    model = load_policy(config,checkpoint_path, device="cpu")

    # -------------------------------------------------------
    # 5. Initialize SFT datasets
    # -------------------------------------------------------

    with open(train_path, "r", encoding="utf-8") as f: 
        data_train = f.read()

    with open(val_path, "r", encoding="utf-8") as f: 
        data_val= f.read()


    #  Create train_dataset and val_dataset

    sft_train_path = compatible_examples(checkpoint_path, train_path, output_path=None)
    SFT_train_dataset = SFTDataset(sft_train_path, config["block_size"], itos, stoi)

    sft_val_path = compatible_examples(checkpoint_path, val_path, output_path=None)
    SFT_val_dataset = SFTDataset(sft_val_path, config["block_size"], itos, stoi)


    

    # -------------------------------------------------------
    # 6. Initialize DataLoaders
    # -------------------------------------------------------

    train_loader = DataLoader(SFT_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SFT_val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------
    # 7. Initialize optimizer and trainer
    # -------------------------------------------------------
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    trainer = TrainSFT(model, num_epochs, optimizer, eval_every, device)

    

    # -------------------------------------------------------
    # 8. Run SFT
    # -------------------------------------------------------

    History = trainer.full_training_loop(train_loader, val_loader)



    # -------------------------------------------------------
    # 9. Create experiment directory
    # -------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path("experiments") / f"sft_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 10. Save training metrics
    # -------------------------------------------------------

    train_losses = History["train_loss"]
    val_losses = History["val_loss"]

    epochs = History.get(
        "Epochs",
        list(range(1, len(train_losses) + 1))
    )

    eval_epochs = History.get(
        "eval_epochs",
        [
            epoch for epoch in epochs
            if epoch % eval_every == 0 or epoch == num_epochs
        ]
    )

    val_by_epoch = dict(zip(eval_epochs, val_losses))

    with open(output_dir / "metrics.csv", "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch, train_loss in zip(epochs, train_losses):

            writer.writerow([
                epoch,
                train_loss,
                val_by_epoch.get(epoch, "")
            ])

    # -------------------------------------------------------
    # 11. Save SFT checkpoint
    # -------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
            "epoch": num_epochs
        },
        output_path
    )

    # -------------------------------------------------------
    # 12. Save training history
    # -------------------------------------------------------

    torch.save(
        History,
        output_dir / "history.pt"
    )

    # -------------------------------------------------------
    # 13. Save learning curves
    # -------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Train"
    )

    plt.plot(
        eval_epochs,
        val_losses,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("SFT cross-entropy loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    print("SFT checkpoint saved to:", output_path)
    print("Experiment saved to:", output_dir)



if __name__ == "__main__":
    main()
