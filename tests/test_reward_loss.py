##############################################################################
#                             Test reward loss 
##############################################################################
import torch 
import pytest
import torch.nn as nn 
from src.losses.reward_loss import RewardLoss

# -------------------------------------------------------
#  Create toy Rewards loss
# -------------------------------------------------------

@pytest.fixture

def reward_loss(): 
    return RewardLoss()

# -------------------------------------------------------
# 1. Test shapes
# -------------------------------------------------------

def test_shape(reward_loss): 

    torch.manual_seed(42)

    chosen_rewards = torch.randn(2)
    rejected_rewards = torch.randn(2)

    loss = reward_loss(chosen_rewards,rejected_rewards)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

# -------------------------------------------------------
# 2. Test Gradient computations 
# -------------------------------------------------------

def test_backward(reward_loss): 

    torch.manual_seed(42)

    chosen_rewards = torch.randn(2, requires_grad=True)
    rejected_rewards = torch.randn(2, requires_grad=True)

    loss = reward_loss(chosen_rewards,rejected_rewards)

    loss.backward()

    for parameter in reward_loss.parameters(): 

        assert torch.isfinite(parameter.grad) 
        assert parameter.grad is not None

# -------------------------------------------------------
# 3. Test loss values 
# -------------------------------------------------------

def test_loss_values(reward_loss): 

    torch.manual_seed(42)

    chosen_rewards = torch.randn(2, requires_grad=True)
    rejected_rewards = torch.randn(2, requires_grad=True)

    loss = reward_loss(chosen_rewards,rejected_rewards)

    loss_computed = - nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

    assert torch.allclose(loss, loss_computed)

# -------------------------------------------------------
# 3. Test loss ordering  
# -------------------------------------------------------

def test_loss_order(reward_loss): 


    chosen_good = torch.tensor([4.0, 3.0])
    rejected_good = torch.tensor([1.0, 2.0])

    chosen_bad = torch.tensor([2.0, 1.0])
    rejected_bad = torch.tensor([2.0, 3.0])

    

    loss_good = reward_loss(chosen_good,rejected_good)
    loss_bad = reward_loss(chosen_bad,rejected_bad)


    assert loss_good < loss_bad