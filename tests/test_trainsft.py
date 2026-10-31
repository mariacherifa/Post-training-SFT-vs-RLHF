
###############################################################
#                 Test SFT Training Function
###############################################################

import math
import torch

from tinygpt.model import TinyGPT
from torch.utils.data import DataLoader, TensorDataset

from src.trainers.training_sft import TrainSFT
from src.data.utils import get_full_text
from src.models.policy import load_policy
from src.data.SFT_dataset import SFTDataset


# ============================================================
# Helper 1: Initialize model and trainer
# ============================================================

def create_trainer(tmp_path, num_epochs=1, lr=1e-3):

    config = {
        "n_embd": 64,
        "vocab_size": 30,
        "num_heads": 4,
        "block_size": 16,
        "num_layers": 2,
        "dropout": 0.0,
    }

    original_model = TinyGPT(**config)

    checkpoint_path = tmp_path / "test_model.pt"

    torch.save(
        {"model_state_dict": original_model.state_dict()},
        checkpoint_path
    )

    model = load_policy(
        config,
        checkpoint_path,
        device="cpu"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    trainer = TrainSFT(
        model=model,
        num_epochs = num_epochs,
        optimizer=optimizer,
        eval_every=1,
        device="cpu"
    )

    return model, trainer, config


# ============================================================
# Helper 2: Create artificial data
# ============================================================

def create_loader(
    config,
    num_examples=4,
    batch_size=2,
    constant_token=None
):

    T = config["block_size"]
    V = config["vocab_size"]

    if constant_token is None:
        tokens = torch.randint(
            0, V, (num_examples, T + 1)
        )
    else:
        tokens = torch.full(
            (num_examples, T + 1),
            constant_token,
            dtype=torch.long
        )

    x = tokens[:, :-1]
    y = tokens[:, 1:]

    loss_mask = torch.zeros(
        (num_examples, T),
        dtype=torch.long
    )

    loss_mask[:, 4:] = 1

    dataset = TensorDataset(x, y, loss_mask)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return loader


# ============================================================
# Test 1: One training epoch updates parameters
# ============================================================

def test_train_one_epoch(tmp_path):

    torch.manual_seed(42)

    model, trainer, config = create_trainer(tmp_path)

    train_loader = create_loader(config)

    initial_weights = [
        p.detach().clone()
        for p in model.parameters()
    ]

    train_loss = trainer.train_one_epoch(train_loader)

    assert math.isfinite(train_loss)

    assert any(
        not torch.equal(w, p)
        for w, p in zip(
            initial_weights,
            model.parameters()
        )
    )


# ============================================================
# Test 2: Evaluation does not update parameters
# ============================================================

def test_evaluate(tmp_path):

    torch.manual_seed(45)

    model, trainer, config = create_trainer(tmp_path)

    val_loader = create_loader(config)

    initial_weights = [
        p.detach().clone()
        for p in model.parameters()
    ]

    val_loss = trainer.evaluate(val_loader)

    assert math.isfinite(val_loss)

    assert all(
        torch.equal(w, p)
        for w, p in zip(
            initial_weights,
            model.parameters()
        )
    )


# ============================================================
# Test 3: Full training loop works
# ============================================================

def test_full_training(tmp_path):

    torch.manual_seed(45)

    model, trainer, config = create_trainer(
        tmp_path,
        num_epochs=3
    )

    train_loader = create_loader(
        config,
        num_examples=4
    )

    val_loader = create_loader(
        config,
        num_examples=2
    )

    history = trainer.full_training_loop(
        train_loader,
        val_loader
    )

    assert history["epoch"] == [1, 2, 3]

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3

    assert all(
        math.isfinite(loss)
        for loss in history["train_loss"]
    )

    assert all(
        math.isfinite(loss)
        for loss in history["val_loss"]
    )


# ============================================================
# Test 4: Model actually learns
# ============================================================

def test_model_learns(tmp_path):

    torch.manual_seed(45)

    model, trainer, config = create_trainer(
        tmp_path,
        num_epochs=30,
        lr=2e-3
    )

    # Tiny dataset with a simple pattern
    train_loader = create_loader(
        config,
        num_examples=2,
        batch_size=2,
        constant_token=5
    )

    # 1. Measure loss BEFORE training
    initial_loss = trainer.evaluate(train_loader)

    # 2. Train repeatedly on the same data
    for epoch in range(30):
        trainer.train_one_epoch(train_loader)

    # 3. Measure loss AFTER training
    final_loss = trainer.evaluate(train_loader)

    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")

    # 4. Check that the model learned
    assert math.isfinite(final_loss)

    assert final_loss < initial_loss, (
        f"Model did not learn: "
        f"initial={initial_loss:.4f}, "
        f"final={final_loss:.4f}"
    )


# ============================================================
# Test 4: Training with SFTDATSet
# ============================================================
    

################################################################
#        Integration Test: SFTDataset + TinyGPT + SFTTrainer
################################################################

import json
import math
import torch
import pytest

from torch.utils.data import DataLoader
from tinygpt.model import TinyGPT

from src.data.SFT_dataset import SFTDataset
from src.data.utils import get_prompt_response, get_full_text
from src.models.policy import load_policy
from src.trainers.training_sft import TrainSFT


# ============================================================
# 1. Create artificial instruction-response dataset
# ============================================================

@pytest.fixture
def sample_data():
    return [
        {
            "instruction": "Say hi",
            "response": "Hello"
        },
        {
            "instruction": "Add 1 and 1",
            "response": "2"
        }
    ]


# ============================================================
# 2. Build toy character-level vocabulary
# ============================================================

def build_vocab(data):

    texts = [get_full_text(example) for example in data]

    chars = sorted(set("".join(texts)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    return itos, stoi


# ============================================================
# 3. Initialize SFTDataset
# ============================================================

@pytest.fixture
def dataset(tmp_path, sample_data):

    # Create temporary JSON dataset
    path = tmp_path / "test_sft.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)

    # Initialize vocabulary
    itos, stoi = build_vocab(sample_data)

    # Character-level tokenizer: reserve enough context
    # to fit the complete prompt-response examples.
    block_size = max(
        len(get_full_text(example))
        for example in sample_data
    )

    dataset = SFTDataset(
        path=path,
        block_size=block_size,
        itos=itos,
        stoi=stoi
    )

    return dataset


# ============================================================
# 4. Test complete SFT training pipeline
# ============================================================

def test_full_training_sft_dataset(tmp_path, dataset):

    torch.manual_seed(45)

    # --------------------------------------------------------
    # Step 1: Model configuration
    # --------------------------------------------------------

    config = {
        "n_embd": 64,
        "vocab_size": len(dataset.stoi),
        "num_heads": 4,
        "block_size": dataset.block_size,
        "num_layers": 2,
        "dropout": 0.0,
    }

    # --------------------------------------------------------
    # Step 2: Initialize TinyGPT
    # --------------------------------------------------------

    original_model = TinyGPT(**config)

    checkpoint_path = tmp_path / "test_model.pt"

    torch.save(
        {"model_state_dict": original_model.state_dict()},
        checkpoint_path
    )

    # --------------------------------------------------------
    # Step 3: Load policy from checkpoint
    # --------------------------------------------------------

    model = load_policy(
        config,
        checkpoint_path,
        device="cpu"
    )

    # Save weights before training
    initial_weights = [
        p.detach().clone()
        for p in model.parameters()
    ]

    # --------------------------------------------------------
    # Step 4: Initialize optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # --------------------------------------------------------
    # Step 5: Check SFTDataset outputs
    # --------------------------------------------------------

    assert len(dataset) == 2

    x, y, loss_mask = dataset[0]

    assert x.shape == (dataset.block_size,)
    assert y.shape == (dataset.block_size,)
    assert loss_mask.shape == (dataset.block_size,)

    assert loss_mask.sum().item() > 0

    # --------------------------------------------------------
    # Step 6: Initialize DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    # --------------------------------------------------------
    # Step 7: Initialize SFTTrainer
    # --------------------------------------------------------

    trainer = TrainSFT(
        model=model,
        optimizer=optimizer,
        num_epochs=3,
        eval_every=1,
        device="cpu"
    )

    # --------------------------------------------------------
    # Step 8: Full training
    # --------------------------------------------------------

    history = trainer.full_training_loop(
        train_loader,
        val_loader
    )

    # --------------------------------------------------------
    # Step 9: Verify history
    # --------------------------------------------------------

    assert history["epoch"] == [1, 2, 3]

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3

    # --------------------------------------------------------
    # Step 10: Verify finite losses
    # --------------------------------------------------------

    assert all(
        math.isfinite(loss)
        for loss in history["train_loss"]
    )

    assert all(
        math.isfinite(loss)
        for loss in history["val_loss"]
    )

    # --------------------------------------------------------
    # Step 11: Verify that model parameters changed
    # --------------------------------------------------------

    assert any(
        not torch.equal(w, p)
        for w, p in zip(
            initial_weights,
            model.parameters()
        )
    )

    print("\nIntegration test passed!")
    print("Training losses:", history["train_loss"])
    print("Validation losses:", history["val_loss"])




