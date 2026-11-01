###############################################################################
#                       Reference policies for RLHF
###############################################################################

from tinygpt.model import TinyGPT
import torch
import copy 

def reference_policy(config, checkpoint_path, device): 
    sft_model = TinyGPT(**config)

    checkpoint = torch.load(checkpoint_path,
                 map_location=device, 
                weights_only= True )
    
    sft_model.load_state_dict(
    checkpoint["model_state_dict"]
)

    ppo_policy = copy.deepcopy(sft_model).to(device)

    ppo_reference = copy.deepcopy(sft_model).to(device)
    ppo_reference.requires_grad_(False)
    ppo_reference.eval()

    return ppo_policy, ppo_reference

