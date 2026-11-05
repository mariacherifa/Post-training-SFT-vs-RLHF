###################################################################
#                        Policy 
###################################################################

import torch 
from tinygpt.model import TinyGPT

def load_policy(config,checkpoint_path, device="cpu"): 

    # Load TinyGPT model

    model = TinyGPT(**config)

    # Loading checkpoints 

    checkpoint = torch.load(checkpoint_path,
                            map_location=device, 
                            weights_only= True )

    # Load previous parameters 

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model.to(device)

