######################################################################
#                   Continued Pretraining TinyGPT
######################################################################

import torch
from tinygpt.model import TinyGPT
from tinygpt.data import encode
from tinygpt.train import Train
from pathlib import Path
from scripts.extend_tinygpt import extend_context

from collections import Counter


# -------------------------------------------------------
# 1. Load extended TinyGPT
# -------------------------------------------------------

def load_extended_model(checkpoint_path, device):

    checkpoints = torch.load(checkpoint_path, 
                             map_location=device, 
                             weights_only=True)

    stoi = checkpoints["stoi"]
    itos = checkpoints["itos"]
    config = checkpoints["config"]

    # Initialize TinyGPT
    model =  TinyGPT(**config)
    # Load pretrained parameters

    model.load_state_dict(
        checkpoints["model_state_dict"]
    )

    model.to(device)

    return model, config, stoi, itos



# -------------------------------------------------------
# 2. Prepare longer sequences
# -------------------------------------------------------

def prepare_data(data_path, stoi, block_size):

    # Read text
    with open(data_path, "r", encoding="utf-8") as f: 
        text = f.read()

    # Encode text
    text_encoded = encode(text, stoi)


    # Split into training and validation
    n = len(text_encoded)
    split = int(n*0.8)

    train_data = text_encoded[:split]
    val_data = text_encoded[split:]

    if len(train_data) <= block_size: 
        raise ValueError("train data is too short")

    if len(val_data) <= block_size: 
        raise ValueError("val data is too short")
    
    
    return torch.tensor(train_data, dtype=torch.long), torch.tensor(val_data, dtype=torch.long)


# -------------------------------------------------------
# 3. Continue pretraining
# -------------------------------------------------------

def train_model(
    model, train_data, val_data, n_embd, batch_size, block_size, num_epochs,optimizer,steps_per_epoch,val_every,device
):

    trainer = Train(model, train_data, val_data, 
                    n_embd, batch_size, block_size,
                    num_epochs,optimizer,
                    steps_per_epoch,val_every,
                    device)

    History = trainer.fit()


    return History


# -------------------------------------------------------
# 4. Save adapted checkpoint
# -------------------------------------------------------

def save_checkpoint(model, config, stoi, itos, path,optimizer):

    checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
    "stoi": stoi,
    "itos": itos
}
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(checkpoint, path)

    
# -------------------------------------------------------
# 5. Extend vocabulary 
# -------------------------------------------------------
def extend_vocab(data_path, stoi, itos): 

    with open(data_path, "r", encoding="utf-8") as f :
        text = f.read()

    new_stoi = stoi.copy()
    new_itos = itos.copy()

    missing_char = set(text) - set(stoi.keys())

    next_id = max(new_stoi.values()) + 1

    for char in sorted(missing_char): 

        new_stoi[char] = next_id
        new_itos[next_id] = char

        next_id += 1

    return new_stoi, new_itos 


# -------------------------------------------------------
# 6. Extend vocabulary model 
# -------------------------------------------------------


def extend_model_vocab(model, config, new_vocab_size):

    old_vocab_size = config["vocab_size"]

    if new_vocab_size < old_vocab_size:
        raise ValueError(
            "New vocabulary cannot be smaller than old vocabulary"
        )

    # 1. Define new model configuration
    new_config = config.copy()
    new_config["vocab_size"] = new_vocab_size

    # 2. Initialize new TinyGPT
    model_new = TinyGPT(**new_config)

    # 3. Get model parameters
    old_state = model.state_dict()
    new_state = model_new.state_dict()

    # 4. Copy pretrained parameters
    with torch.no_grad():

        for name in new_state:

            if name not in old_state:
                raise ValueError(f"Missing parameter: {name}")

            old_param = old_state[name]
            new_param = new_state[name]

            # Parameters with identical shapes
            if old_param.shape == new_param.shape:

                new_param.copy_(old_param)

            # Vocabulary-dependent parameters
            elif (
                old_param.ndim >= 1
                and old_param.shape[0] == old_vocab_size
                and new_param.shape[0] == new_vocab_size
                and old_param.shape[1:] == new_param.shape[1:]
            ):

                new_param[:old_vocab_size].copy_(old_param)

                print(f"Extended parameter: {name}")

            else:
                raise ValueError(
                    f"Unexpected shape mismatch for {name}: "
                    f"{old_param.shape} -> {new_param.shape}"
                )

    # 5. Load transferred parameters
    model_new.load_state_dict(new_state)

    return model_new


# -------------------------------------------------------
# 7. Main experiment
# -------------------------------------------------------


def main():

    # -------------------------------------------------------
    # 1. Define paths
    # -------------------------------------------------------

    original_checkpoint_path = (
        Path.home()
        / "Desktop/Tiny_GPT_github/experiments"
        / "tinystories_data_longer_training_more_data_20261103_110730"
        / "last_model.pt"
    )

    data_path = (
        Path.home()
        / "Desktop/Tiny_GPT_github/data/TinyStories-train.txt"
    )

    extended_checkpoint_path = Path(
        "checkpoints/tinygpt_extended.pt"
    )

    output_checkpoint_path = Path(
        "checkpoints/tinygpt_2048_pretrained.pt"
    )

    history_path = Path(
        "experiments/pretraining_2048/history.pt"
    )

    # -------------------------------------------------------
    # 2. Define hyperparameters
    # -------------------------------------------------------

    new_block_size = 2048

    batch_size = 1
    learning_rate = 1e-4
    num_epochs = 2
    steps_per_epoch = 20
    val_every = 1

    device = "cpu"

    torch.manual_seed(42)

    # -------------------------------------------------------
    # 3. Check paths
    # -------------------------------------------------------

    if not original_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {original_checkpoint_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # -------------------------------------------------------
    # 4. Load original checkpoint
    # -------------------------------------------------------

    print("Checkpoint path:", original_checkpoint_path)
    print("File exists:", original_checkpoint_path.exists())
    print("File size:", original_checkpoint_path.stat().st_size)

    checkpoint = torch.load(
        original_checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    full_config = checkpoint["config"]

    # Keep only TinyGPT architecture parameters
    config = {
        "n_embd": full_config["n_embd"],
        "vocab_size": len(stoi),
        "num_heads": full_config["num_heads"],
        "block_size": full_config["block_size"],
        "num_layers": full_config["num_layers"],
        "dropout": full_config["dropout"]
    }

    # -------------------------------------------------------
    # 5. Extend TinyGPT vocabulary
    # -------------------------------------------------------

    new_stoi, new_itos = extend_vocab(
        data_path,
        stoi,
        itos
    )

    print("Original vocabulary size:", len(stoi))
    print("Extended vocabulary size:", len(new_stoi))

    # -------------------------------------------------------
    # 6. Extend TinyGPT context
    # -------------------------------------------------------

    model = extend_context(
        config,
        new_block_size=new_block_size
    )

    # Update configuration after context extension
    new_config = config.copy()
    new_config["block_size"] = new_block_size

    # -------------------------------------------------------
    # 7. Extend TinyGPT model vocabulary
    # -------------------------------------------------------

    model = extend_model_vocab(
        model,
        new_config,
        new_vocab_size=len(new_stoi)
    )

    new_config["vocab_size"] = len(new_stoi)

    model = model.to(device)

    print("New block size:", new_config["block_size"])
    print("New vocabulary size:", new_config["vocab_size"])

    # -------------------------------------------------------
    # 8. Save extended checkpoint
    # -------------------------------------------------------

    extended_checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    extended_checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": new_config,
        "stoi": new_stoi,
        "itos": new_itos
    }

    torch.save(
        extended_checkpoint,
        extended_checkpoint_path
    )

    print("Extended checkpoint saved.")

    # -------------------------------------------------------
    # 9. Prepare training and validation data
    # -------------------------------------------------------

    train_data, val_data = prepare_data(
        data_path,
        new_stoi,
        block_size=new_block_size
    )

    print("Training tokens:", len(train_data))
    print("Validation tokens:", len(val_data))

    # Verify token IDs are compatible with the model
    assert int(train_data.max()) < new_config["vocab_size"]
    assert int(val_data.max()) < new_config["vocab_size"]

    # -------------------------------------------------------
    # 10. Initialize optimizer
    # -------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    # -------------------------------------------------------
    # 11. Continue pretraining
    # -------------------------------------------------------

    History = train_model(
        model,
        train_data,
        val_data,
        new_config["n_embd"],
        batch_size,
        new_block_size,
        num_epochs,
        optimizer,
        steps_per_epoch,
        val_every,
        device
    )

    # -------------------------------------------------------
    # 12. Save adapted TinyGPT checkpoint
    # -------------------------------------------------------

    output_checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    adapted_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": new_config,
        "stoi": new_stoi,
        "itos": new_itos,
        "num_epochs": num_epochs
    }

    torch.save(
        adapted_checkpoint,
        output_checkpoint_path
    )

    print("Adapted checkpoint saved.")

    # -------------------------------------------------------
    # 13. Save training history
    # -------------------------------------------------------

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(History, history_path)

    print("Training history saved.")

    return History


if __name__ == "__main__":
    main()



