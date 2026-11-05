
################################################################
#                 Extend TinyGPT Context
################################################################

import torch
import torch.nn as nn

from tinygpt.model import TinyGPT


def extend_context(config, new_block_size):

    #  Initialize the original TinyGPT
    model = TinyGPT(**config)

    old_block_size = config["block_size"]
    
    new_config = config.copy()
    new_config["block_size"] = new_block_size

    model_new = TinyGPT(**new_config)


    #  Copy pretrained parameters
    old_state = model.state_dict()
    new_state = model_new.state_dict()

    pos_key = "pos_embd.weight"

    if pos_key not in old_state or pos_key not in new_state:
        raise KeyError("Positional embedding key not found")
    #  Extend positional embeddings

    for name in new_state:

        if name == pos_key:

            # Copy only pretrained positional embeddings
            new_state[name][:old_block_size] = old_state[name]

        elif name in old_state:

            if old_state[name].shape == new_state[name].shape:

                new_state[name] = old_state[name].clone()

            else:

                # Causal attention masks depend on block_size.
                # Keep the newly initialized larger buffers.
                if name in dict(model_new.named_buffers()):
                    continue

                raise ValueError(
                    f"Unexpected shape mismatch: {name}"
                )

    
    # Load transferred parameters

    model_new.load_state_dict(new_state)

    
    # 7. Verify positional embeddings
    
    new_positions = model_new.state_dict()[pos_key]
    old_positions = old_state[pos_key]

    assert new_positions.shape[0] == new_block_size

    assert torch.equal(
        new_positions[:old_block_size],
        old_positions
    )

    print(
        f"TinyGPT context extended from "
        f"{old_block_size} to {new_block_size}"
    )

    return model_new


