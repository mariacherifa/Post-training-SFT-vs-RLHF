####################################################################
#                   Test loss SFT 
####################################################################

import torch 
import torch.nn as nn
import pytest 
from src.losses.sft_loss import SFTLoss


torch.manual_seed(42)

logits = torch.randn((2, 4, 3), dtype=torch.float32)
targets = torch.randint(0, 3, (2,4), dtype=torch.long)
loss_mask = torch.tensor([
    [0, 0, 1, 1],
    [0, 1, 1, 0]
], dtype=torch.float)


# Test shapes

def test_shapes(): 
    loss_model = SFTLoss()

    loss_sft = loss_model(logits, targets, loss_mask)

    assert loss_sft.ndim == 0
    assert loss_sft.numel() == 1

# test loss without mask 
def test_without_mask(): 
    loss_model = SFTLoss()
    loss_sft = loss_model(logits, targets, loss_mask = None)

    B, T, V = logits.shape

    expected_loss = nn.functional.cross_entropy(
        logits.reshape(B*T, V),
        targets.reshape(B*T)
    )


    torch.testing.assert_close(loss_sft, expected_loss)


# Test : masking works 

def test_masked_loss():

    loss_model = SFTLoss()

    loss_sft = loss_model(logits, targets, loss_mask)

    B, T, V = logits.shape

    # Compute the loss for every token
    loss_per_token = nn.functional.cross_entropy(
        logits.reshape(B*T, V),
        targets.reshape(B*T),
        reduction="none"
    )

    # Recover shape (B,T)
    loss_per_token = loss_per_token.reshape(B, T)

    # Compute expected masked loss
    expected_loss = (
        loss_per_token * loss_mask
    ).sum() / loss_mask.sum()

    torch.testing.assert_close(loss_sft, expected_loss)


# test : gradient computations 

def test_backward():
    # Use fresh logits to isolate this test
    test_logits = torch.randn(
        (2, 4, 3),
        requires_grad=True
    )

    loss_model = SFTLoss()

    loss = loss_model(test_logits, targets, loss_mask)

    # Compute gradients
    loss.backward()

    # Verify gradients exist
    assert test_logits.grad is not None

    # Verify masked positions have zero gradients
    assert torch.all(
        test_logits.grad[loss_mask == 0] == 0
    )

    # Verify supervised positions receive gradients
    assert torch.any(
        test_logits.grad[loss_mask == 1] != 0
    )

