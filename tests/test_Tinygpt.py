###########################################################################
#                           Tests TinyGPT model
###########################################################################

import torch 
import pytest 
from src.models.policy import load_policy
from tinygpt.model import TinyGPT


# -------------------------------------------------------
# 1. TinyGPT well imported
# -------------------------------------------------------
device="cpu"

def test_load_policy(tmp_path): 

     # 1. Define a small model configuration
    config = {
        "n_embd": 64,
        "vocab_size": 30,
        "num_heads": 4,
        "block_size": 16,
        "num_layers": 2,
        "dropout": 0.1,
    }

    # 2. Create a TinyGPT model
    original_model = TinyGPT(**config)

    # 3. Save a temporary checkpoint
    checkpoint_path = tmp_path / "test_model.pt"

    torch.save(
        {"model_state_dict": original_model.state_dict()},
        checkpoint_path
    )

    # 4. Load the model using your function
    loaded_model = load_policy(
        config,
        checkpoint_path,
        device="cpu"
    )

    # 5. Verify the model type
    assert isinstance(loaded_model, TinyGPT)

    # 6. Verify that all weights are identical
    for name, param in original_model.state_dict().items():
        assert torch.equal(
            param,
            loaded_model.state_dict()[name]
        )

    # 7. Verify the device
    assert next(loaded_model.parameters()).device.type == "cpu"



