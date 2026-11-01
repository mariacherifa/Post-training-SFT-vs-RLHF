##############################################################################
#                             Test reward model
##############################################################################

import torch 
from src.models.reward_model import RewardModel
from tinygpt.transformer import BlockTransformer
import pytest
from tinygpt.data import build_vocab
import math


# -------------------------------------------------------
#  Create a reward model 
# -------------------------------------------------------
@pytest.fixture
def reward_model():
    text = "Hello Maria how are you"
    _, vocab_size, _, _ = build_vocab(text)

    n_embd = 64
    num_heads = 4
    block_size = 128
    dropout = 0.1
    num_layer = 2

    model = RewardModel(
        n_embd,
        num_heads,
        block_size,
        vocab_size,
        dropout,
        num_layer
    )

    return model

# -------------------------------------------------------
#  Create toy example 
# -------------------------------------------------------

@pytest.fixture
def sample_inputs():

    ids = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0],
        [2, 3, 4, 5, 6, 0, 0, 0]
    ])

    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0]
    ], dtype=torch.bool)

    return ids, attention_mask

# -------------------------------------------------------
# Test 1: check the reward is scalar
# -------------------------------------------------------

def test_scalar(reward_model, sample_inputs): 

    ids, attention_mask = sample_inputs

    reward_model.eval()

    with torch.no_grad():
        rewards = reward_model(ids, attention_mask)

    assert rewards.shape == (2,)


# -------------------------------------------------------
# Test 2: reward is finite
# -------------------------------------------------------

def test_finite(reward_model, sample_inputs): 

    ids, attention_mask = sample_inputs

    reward_model.eval()

    with torch.no_grad():
        rewards = reward_model(ids, attention_mask)

    assert torch.all(torch.isfinite(rewards))
    assert rewards is not None


# -------------------------------------------------------
# Test 2: Gradient computation 
# -------------------------------------------------------

def test_backward(reward_model, sample_inputs): 

    ids, attention_mask = sample_inputs

    rewards = reward_model(ids, attention_mask)

    loss = rewards.mean()
    
    loss.backward()

    for parameter in reward_model.parameters():

        assert parameter.grad is not None 

        assert torch.isfinite(parameter.grad).all()
