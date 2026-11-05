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
#  Create toy example 
# -------------------------------------------------------

@pytest.fixture
def sample_inputs():

    ids = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0],
        [2, 3, 4, 5, 6, 0, 0, 0] ])
    

    return ids

# -------------------------------------------------------
#  Test value model 
# -------------------------------------------------------

def test_value_model(value_model, sample_inputs): 

    model = value_model
    values = model(sample_inputs)

    assert torch.isfinite(values).all()
    assert values.shape == (2,8)

    values.sum().backward()

    for p in model.parameters(): 
        assert p.grad is not None 
        assert torch.isfinite(p.grad).all()
    