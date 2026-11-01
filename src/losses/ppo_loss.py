######################################################################
#                              PPO objective 
######################################################################

import torch

def PPO_Loss(new_logprobs, old_logprobs, advantages, epsilon): 

    rho_t = torch.exp(new_logprobs - old_logprobs)

    clipped_ratio = torch.clamp(
    rho_t,
    min=1 - epsilon,
    max=1 + epsilon)
    
    ppo_objective = torch.min(rho_t* advantages , clipped_ratio*advantages).mean()

    return - ppo_objective