#########################################################################
#                              Training SFT
#########################################################################

import torch 
import torch.nn as nn
from src.models.policy import load_policy
from src.losses.sft_loss import SFTLoss

class TrainSFT(nn.Module): 
    def __init__(self, model, num_epochs, optimizer, eval_every, device):
        super().__init__()

        self.optimizer = optimizer 
        self.num_epochs = num_epochs
        self.eval_every = eval_every
        self.loss_fn = SFTLoss()
        self.device = torch.device(device) 
        self.model = model.to(self.device) 

    # Training one epoch 

    def train_one_epoch(self,train_loader): 

        self.model.train()

        total_loss = 0.0 
        total_tokens = 0

        for  x,y, loss_mask in train_loader: 
            
            x = x.to(self.device)
            y = y.to(self.device)
            loss_mask = loss_mask.to(self.device)

            self.optimizer.zero_grad()


            _, logits = self.model(x,y)


            loss = self.loss_fn(logits, y, loss_mask)

            loss.backward()

            self.optimizer.step()

            n_tokens = loss_mask.sum().item()

            total_loss += loss.item() * n_tokens

            total_tokens += n_tokens

        total_avg_loss = total_loss/total_tokens

        return total_avg_loss


    # Evaluation function 

    def evaluate(self,val_loader): 

            self.model.eval()
            total_loss = 0.0 
            total_tokens = 0

            with torch.no_grad():
                 for x,y, loss_mask in val_loader: 

                    x = x.to(self.device)
                    y = y.to(self.device)
                    loss_mask = loss_mask.to(self.device)
                        
    
                    _, logits = self.model(x,y)

    
                    loss = self.loss_fn(logits, y, loss_mask)

                    n_tokens = loss_mask.sum().item()
                        
                    total_loss += loss.item() * loss_mask.sum().item()

                    total_tokens += n_tokens
    
            total_avg_loss = total_loss/total_tokens
            return total_avg_loss

    # Training for many epochs

    def full_training_loop(self, train_loader, val_loader): 

        # History to store relevant metrics 

        History = {"epoch": [], "train_loss":[], "val_loss":[]}

        for epoch in range(self.num_epochs): 

            train_loss = self.train_one_epoch(train_loader)

            if (epoch + 1)% self.eval_every == 0 or epoch == self.num_epochs -1: 
                val_loss = self.evaluate(val_loader)

                History["epoch"].append(epoch+1)
                History["train_loss"].append(train_loss)
                History["val_loss"].append(val_loss)

                print(f"epochs : [{epoch +1}/{self.num_epochs}]"
                      f"train_loss {train_loss: .4f}"
                      f"val_loss {val_loss: .4f}")

        return History









        
    


