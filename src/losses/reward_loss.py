###############################################################################
#                                Reward Loss 
###############################################################################

import torch 
import torch.nn as nn 

class RewardLoss(nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self, reward_chosen, reward_rejected): 

        if reward_chosen is None or reward_rejected is None:
            raise ValueError("Rewards cannot be None.")
        
        if reward_chosen.shape != reward_rejected.shape:
            raise ValueError("Reward shapes must match.")

        loss = -nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()

        return loss 


