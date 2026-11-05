################################################################
#                    SFT loss 
################################################################
 
import torch.nn as nn


class SFTLoss(nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, loss_mask=None): 
  
        if loss_mask is None: 
            if targets is None: 
                loss = None 

            else: 
                B,T,C = logits.shape   
                logits_flat = logits.reshape(B*T, C)
                target_flat = targets.reshape(B*T)
                loss = nn.functional.cross_entropy(logits_flat, target_flat)

        else: 
            B,T,C = logits.shape 
            logits_flat = logits.reshape(B*T, C)
            target_flat = targets.reshape(B*T)
            loss_mask_flat = loss_mask.reshape(B*T).float()

            # compute cross entropy without avereging losses 
            
            loss_per_token = nn.functional.cross_entropy(logits_flat,target_flat,reduction="none") 
            denominator = loss_mask_flat.sum()


            if denominator.item() == 0:
                raise ValueError("No supervised tokens in the batch.")
            
            loss = (loss_per_token * loss_mask_flat ).sum() / denominator

        return loss 


        

    

