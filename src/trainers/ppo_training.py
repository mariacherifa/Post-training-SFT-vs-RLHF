
######################################################################
#                        Train reward
######################################################################

import torch
import torch.nn as nn
from losses.ppo_loss import PPO_Loss
from data.utils import token_log_probs
from models.advantages import cumulative_future_returns
from models.rollout import generate_rollout


class PPOTrain:

    def __init__(
        self,
        ppo_policy,
        ppo_reference,
        reward_model,
        value_model,
        optimizer,
        block_size,
        max_token,
        beta,
        epsilon,
        gamma,
        device
    ):
        self.ppo_policy = ppo_policy
        self.ppo_reference = ppo_reference
        self.reward_model = reward_model
        self.value_model = value_model

        self.optimizer = optimizer

        self.block_size = block_size
        self.max_token = max_token
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

    def compute_new_logprobs(self, full_ids, max_token):

        logits, _ = self.ppo_policy(full_ids)

        new_logprobs = token_log_probs(logits, full_ids)

        return new_logprobs[:, -max_token:]

    def train_one_epoch(self, train_loader):

        # Keep dropout disabled for consistent PPO log-probabilities.
        # eval() does not disable gradients.
        self.ppo_policy.eval()
        self.ppo_reference.eval()
        self.reward_model.eval()
        self.value_model.eval()

        all_losses = 0.0
        num_prompts = 0

        for prompt_ids in train_loader:

            prompt_ids = prompt_ids.to(self.device)
            P = prompt_ids.shape[1]

            # 1. Collect rollout
            with torch.no_grad():

                full_ids, old_logprobs, ref_logprobs, token_rewards = generate_rollout(
                    self.ppo_policy,
                    self.ppo_reference,
                    self.reward_model,
                    prompt_ids,
                    self.block_size,
                    self.max_token,
                    self.beta
                )

                # 2. Compute returns
                returns = cumulative_future_returns(
                    token_rewards,
                    gamma=self.gamma
                )

                # 3. Estimate advantages
                old_values = self.value_model(full_ids)[:, P - 1:-1]
                advantages = returns - old_values

            # 4. Recompute log-probabilities with gradients
            new_logprobs = self.compute_new_logprobs(
                full_ids,
                self.max_token
            )

            # 5. Policy loss
            policy_loss = PPO_Loss(
                new_logprobs,
                old_logprobs,
                advantages,
                self.epsilon
            )

            # 6. Critic loss
            values = self.value_model(full_ids)[:, P - 1:-1]

            value_loss = nn.functional.mse_loss(
                values,
                returns.detach()
            )

            # 7. Optimize both models
            total_loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            all_losses += total_loss.item()
            num_prompts += 1

        total_avg_loss = all_losses / num_prompts

        return total_avg_loss

    def evaluate(self, val_loader):

        # Keep dropout disabled for consistent PPO log-probabilities.
        # eval() does not disable gradients.
        self.ppo_policy.eval()
        self.ppo_reference.eval()
        self.reward_model.eval()
        self.value_model.eval()

        all_losses = 0.0
        num_prompts = 0

        for prompt_ids in val_loader:

            prompt_ids = prompt_ids.to(self.device)
            P = prompt_ids.shape[1]

            # 1. Collect rollout
            with torch.no_grad():

                full_ids, old_logprobs, ref_logprobs, token_rewards = generate_rollout(
                    self.ppo_policy,
                    self.ppo_reference,
                    self.reward_model,
                    prompt_ids,
                    self.block_size,
                    self.max_token,
                    self.beta
                )

                # 2. Compute returns
                returns = cumulative_future_returns(
                    token_rewards,
                    gamma=self.gamma
                )

                # 3. Estimate advantages
                old_values = self.value_model(full_ids)[:, P - 1:-1]
                advantages = returns - old_values

            # 4. Recompute log-probabilities with gradients
            new_logprobs = self.compute_new_logprobs(
                full_ids,
                self.max_token
            )

            # 5. Policy loss
            policy_loss = PPO_Loss(
                new_logprobs,
                old_logprobs,
                advantages,
                self.epsilon
            )

            # 6. Critic loss
            values = self.value_model(full_ids)[:, P - 1:-1]

            value_loss = nn.functional.mse_loss(
                values,
                returns.detach()
            )

            # 7. Optimize both models
            total_loss = policy_loss + 0.5 * value_loss

            all_losses += total_loss.item()
            num_prompts += 1

        total_avg_loss = all_losses / num_prompts

        return total_avg_loss

    def full_training(
        self,
        train_loader,
        val_loader,
        num_epochs,
        eval_every=1
    ):

        history = {
            "train_loss": [],
            "val_loss": [],
            "eval_epochs": []
        }

        for epoch in range(1, num_epochs + 1):

            # Train for one epoch
            train_loss = self.train_one_epoch(train_loader)

            history["train_loss"].append(train_loss)

            # Evaluate periodically
            if epoch % eval_every == 0 or epoch == num_epochs:

                val_loss = self.evaluate(val_loader)

                history["val_loss"].append(val_loss)
                history["eval_epochs"].append(epoch)

                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

            else:
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f}"
                )

        return history
