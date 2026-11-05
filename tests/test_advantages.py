######################################################################
#                     Test advantages computations 
######################################################################
import torch 
import pytest 
from src.models.advantages import cumulative_future_returns, advantage

##############################################################################
#                             Test value model
##############################################################################

import torch 
import pytest 
from src.models.value_model import ValueModel
from tinygpt.data import build_vocab

# -------------------------------------------------------
#  Create a value model 
# -------------------------------------------------------
@pytest.fixture

def value_model(): 
    text = "Hello Maria how are you"
    _, vocab_size, _, _ = build_vocab(text)

    n_embd = 64
    num_heads = 4
    block_size = 128
    vocab_size = 30
    dropout = 0.1
    num_layer = 2

    model = ValueModel(n_embd, num_heads, block_size,vocab_size, dropout, num_layer)

    return model

# -------------------------------------------------------
#  Test cumulative_future_returns
# -------------------------------------------------------
def test_cumulative_future_returns(): 

    torch.manual_seed(42)

    token_rewards = torch.randint(1,5,(2,10))

    gamma = 1

    returns = cumulative_future_returns(token_rewards, gamma)

    assert torch.isfinite(returns).all()
    assert returns.shape == (2,10)

# -------------------------------------------------------
#  Test advantages function
# -------------------------------------------------------
def test_advantages(value_model): 

    torch.manual_seed(42)
     
    full_ids = torch.randint(1,5, (2,15))
    token_rewards = torch.randint(1,5, (2,5))
    gamma = 1 

    advantages_vals = advantage(value_model, full_ids, token_rewards, gamma=1)

    assert torch.isfinite(advantages_vals).all()
    assert advantages_vals.shape == (2,5)
    assert not advantages_vals.requires_grad
    

