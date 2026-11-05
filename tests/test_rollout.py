
##############################################################################
#                             Rollout generation test
##############################################################################

import torch
import pytest
import copy

from src.models.rollout import generate_next_token, generate_rollout, rewards_sampled_trajectories
from src.data.utils import kl_penality, token_log_probs
from tinygpt.model import TinyGPT
from src.models.reward_model import RewardModel


##############################################################################
#                             Create toy example
##############################################################################

@pytest.fixture
def models():

    config = {
        "n_embd": 64,
        "vocab_size": 30,
        "num_heads": 4,
        "block_size": 16,
        "num_layers": 2,
        "dropout": 0.1,
    }

    sft_model = TinyGPT(**config)

    ppo_policy = copy.deepcopy(sft_model)
    ppo_reference = copy.deepcopy(sft_model)

    ppo_reference.requires_grad_(False)
    ppo_reference.eval()

    return ppo_policy, ppo_reference


##########################################################################
#  Test generate_next_token function
##########################################################################

def test_generate_next_token(models):

    torch.manual_seed(42)

    encoded_prompt = torch.randint(
        1, 12, (1, 8)
    )

    max_token = 5
    block_size = 16

    full_ids, old_logprobs = generate_next_token(
        models[0],
        encoded_prompt,
        block_size,
        max_token
    )

    # Check full sequence dimensions
    assert full_ids.shape == (1, 13)

    # Check sampled token log-probabilities
    assert old_logprobs.shape == (1, 5)

    # Check token type
    assert full_ids.dtype == torch.long

    # Check log-probabilities are finite
    assert torch.isfinite(old_logprobs).all()

    # Check generated sequence starts with the original prompt
    assert torch.equal(
        full_ids[:, :8],
        encoded_prompt
    )

    # Check generated tokens belong to vocabulary
    assert torch.all(full_ids >= 0)
    assert torch.all(full_ids < 30)


##########################################################################
#  Test generate_rollout function
##########################################################################
def test_generate_rollout(models): 

    torch.manual_seed(42)
    
    encoded_prompt = torch.randint(
    1, 12, (1, 8))
    
    max_token = 5
    block_size = 16
    
    full_ids, old_logprobs = generate_next_token(
            models[0],
            encoded_prompt,
            block_size,
            max_token
        )

    block_size = 16
    max_token = 5 
    beta = 0.3

    full_ids, old_logprobs, ref_logprobs, kl_pena = generate_rollout(models[0],models[1], 
                                                                     encoded_prompt, 
                                                                     block_size, 
                                                                     max_token, 
                                                                     beta)

    # full_ids tests

    assert full_ids.shape == (1,13)
    assert full_ids.dtype == torch.long
    assert torch.all(full_ids >= 0)
    assert torch.all(full_ids < 30)
    assert torch.equal(full_ids[:,:8], encoded_prompt)

    # old_logprobs tests 
        
    assert old_logprobs.shape == (1,5)
    assert torch.isfinite(old_logprobs).all()
    
    # ref_logprobs tests
    
    assert torch.isfinite(ref_logprobs).all()
    assert ref_logprobs.shape == (1,5)
    
    # kl_pena tests
    
    assert torch.isfinite(kl_pena).all()
    assert kl_pena.shape == (1,5)

##########################################################################
#  Test rewards_sampled_trajectories
##########################################################################

def test_rewards_sampled_trajectories(models): 

    torch.manual_seed(42)
    
    encoded_prompt = torch.randint(
    1, 12, (1, 8))
    
    max_token = 5
    block_size = 16
    

    block_size = 16
    max_token = 5 
    beta = 0.3

    _, _, _, kl_pena = generate_rollout(models[0],models[1], 
                                            encoded_prompt, 
                                            block_size, 
                                            max_token, 
                                            beta)
    
    reward_model = RewardModel(n_embd = 64,num_heads = 4, block_size=16, vocab_size= 30, dropout=0.1, num_layer=2)
            

    full_ids, old_logprobs,ref_logprobs, token_rewards = rewards_sampled_trajectories(models[0],models[1], 
                                                                     reward_model,
                                                                         encoded_prompt,
                                                                         block_size,
                                                                         max_token,
                                                                         beta)

    # full_ids tests

    assert full_ids.shape == (1,13)
    assert full_ids.dtype == torch.long
    assert torch.all(full_ids >= 0)
    assert torch.all(full_ids < 30)
    assert torch.equal(full_ids[:,:8], encoded_prompt)

    # old_logprobs tests 
        
    assert old_logprobs.shape == (1,5)
    assert torch.isfinite(old_logprobs).all()
    
    # ref_logprobs tests
    
    assert torch.isfinite(ref_logprobs).all()
    assert ref_logprobs.shape == (1,5)
    
    # kl_pena tests
    assert torch.isfinite(token_rewards).all()
    assert token_rewards.shape == (1,5)
    assert not torch.equal(token_rewards, kl_pena)
    assert torch.allclose(
    token_rewards[:, :-1],
    kl_pena[:, :-1],
    atol=1e-6,
    rtol=0)

    assert not token_rewards.requires_grad

    


    




