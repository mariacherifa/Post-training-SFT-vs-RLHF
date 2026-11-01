##############################################################################
#                             Test training reward
##############################################################################
import torch 
import pytest
import torch.nn as nn 
from src.trainers.train_reward import TrainReward
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------
# 1. Create a toy reward model
# -------------------------------------------------------

class ToyRewardModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(16, 8)
        self.reward_head = nn.Linear(8, 1)

    def forward(self, input_ids, attention_mask):

        x = self.embedding(input_ids)

        mask = attention_mask.unsqueeze(-1).float()

        # Average only valid token embeddings
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        rewards = self.reward_head(x).squeeze(-1)

        return rewards


# -------------------------------------------------------
# 2. Create an artificial DataLoader
# -------------------------------------------------------

@pytest.fixture
def train_loader():

    chosen_ids = torch.tensor([
        [1, 2, 3, 0],
        [1, 4, 0, 0],
        [2, 3, 4, 0],
        [5, 6, 7, 0],
        [1, 2, 4, 0]
    ], dtype=torch.long)

    rejected_ids = torch.tensor([
        [1, 8, 3, 0],
        [1, 9, 0, 0],
        [2, 7, 4, 0],
        [5, 8, 7, 0],
        [1, 9, 4, 0]
    ], dtype=torch.long)

    chosen_mask = chosen_ids != 0
    rejected_mask = rejected_ids != 0

    dataset = TensorDataset(
        chosen_ids,
        rejected_ids,
        chosen_mask,
        rejected_mask
    )

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )


# -------------------------------------------------------
# 3. Initialize the trainer
# -------------------------------------------------------

@pytest.fixture
def trainer():

    torch.manual_seed(42)

    model = ToyRewardModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    return TrainReward(
        model=model,
        num_epochs=3,
        optimizer=optimizer,
        val_every=2,
        device="cpu"
    )


# -------------------------------------------------------
# Test one epoch training
# -------------------------------------------------------

def test_one_epoch_train(trainer, train_loader):

    model = trainer.model

    # Save parameters before training
    params_before = [
        p.detach().clone()
        for p in model.parameters()
    ]

    # Train for one epoch
    loss = trainer.train_one_epoch(train_loader)

    # Verify the loss
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))

    # Verify that parameters were updated
    params_after = list(model.parameters())

    changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(params_before, params_after)
    )

    assert changed



# -------------------------------------------------------
# Test evaluation 
# -------------------------------------------------------


def test_evaluation(trainer, train_loader):

    model = trainer.model

    # Save parameters before training
    params_before = [
        p.detach().clone()
        for p in model.parameters()
    ]

    # Train for one epoch
    loss = trainer.evaluate(train_loader)

    # Verify the loss
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))

    # essert evaluation mode 

    assert not model.training

    # Verify that parameters were updated
    params_after = list(model.parameters())

    changed = all(
        torch.equal(before, after.detach())
        for before, after in zip(params_before, params_after)
    )

    assert changed


# -------------------------------------------------------
# Test full training 
# -------------------------------------------------------


def test_full_train(trainer, train_loader):

    history = trainer.full_train(
        train_loader,
        train_loader
    )

    # Check returned dictionary
    assert set(history.keys()) == {
        "epoch",
        "train_loss",
        "val_loss"
    }

    # num_epochs=3, val_every=2
    assert history["epoch"] == [2, 3]

    # Check number of recorded losses
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2

    # Check losses are finite
    for loss in history["train_loss"]:
        assert torch.isfinite(torch.tensor(loss))

    for loss in history["val_loss"]:
        assert torch.isfinite(torch.tensor(loss))


