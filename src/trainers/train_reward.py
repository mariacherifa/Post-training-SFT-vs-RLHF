######################################################################
#                        Train reward 
######################################################################

import torch 
import torch.nn as nn
from src.losses.reward_loss import RewardLoss


class TrainReward(nn.Module): 
    def __init__(self,model, num_epochs, optimizer, val_every,device):
        super().__init__()

        self.model = model
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.val_every = val_every
        self.device = device
        self.criterion = RewardLoss()

    def train_one_epoch(self, train_loader): 
        self.model.train()

        total_loss = 0.0
        total_samples = 0


        for chosen_ids, rejected_ids, chosen_mask, rejected_mask in train_loader:

            chosen_ids = chosen_ids.to(self.device)
            rejected_ids = rejected_ids.to(self.device)
            chosen_mask = chosen_mask.to(self.device)
            rejected_mask = rejected_mask.to(self.device)

            chosen_rewards = self.model(chosen_ids,chosen_mask)
            rejected_rewards = self.model(rejected_ids,rejected_mask)

            self.optimizer.zero_grad()
    
            loss = self.criterion(chosen_rewards, rejected_rewards)

            loss.backward()
            self.optimizer.step()

            batch_size = chosen_ids.size(0)

            total_loss += loss.item()*batch_size
            total_samples += batch_size



        total_avg_loss = total_loss / total_samples

        return  total_avg_loss

    def evaluate(self,val_loader): 
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0 

        with torch.no_grad():
            for chosen_ids, rejected_ids, chosen_mask, rejected_mask in val_loader:

                chosen_ids = chosen_ids.to(self.device)
                rejected_ids = rejected_ids.to(self.device)
                chosen_mask = chosen_mask.to(self.device)
                rejected_mask = rejected_mask.to(self.device)

                chosen_rewards = self.model(chosen_ids, chosen_mask)
                rejected_rewards = self.model(rejected_ids, rejected_mask)

                loss = self.criterion(chosen_rewards, rejected_rewards)

                batch_size = chosen_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        total_avg_loss = total_loss / total_samples
        
        return  total_avg_loss

    def full_train(self, train_loader, val_loader): 

        History = {"epoch": [], "train_loss":[], "val_loss":[]}

        for epoch in range(self.num_epochs): 
            train_loss = self.train_one_epoch(train_loader)

            if (epoch + 1)% self.val_every == 0 or epoch == self.num_epochs - 1:
                val_loss = self.evaluate(val_loader)

                History["epoch"].append(epoch + 1)
                History["train_loss"].append(train_loss)
                History["val_loss"].append(val_loss)


        return History  




