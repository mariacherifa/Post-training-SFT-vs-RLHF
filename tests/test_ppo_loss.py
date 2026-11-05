####################################################################
#                   Test loss PPO 
####################################################################

import torch 
import torch.nn as nn
import pytest 
from src.losses.ppo_loss import PPO_Loss

def test_ppo_loss(): 
    torch.manual_seed(42)

    new_logprobs = torch.randn((2,10))
    old_logprobs = torch.randn((2,10))
    advantages = torch.randint(1,2, (2,10))
    epsilon = 0.1

    ppo_loss = PPO_Loss(new_logprobs, old_logprobs, advantages, epsilon)

    assert torch.isfinite(ppo_loss).all()
    assert ppo_loss is not None 
    