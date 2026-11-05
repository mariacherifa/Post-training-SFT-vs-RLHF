
######################################################################
#                         PPO Training
######################################################################

import torch
import torch.nn.functional as F

from src.losses.ppo_loss import PPO_Loss
from src.data.utils import token_log_probs
from src.models.advantages import cumulative_future_returns
from src.models.rollout import generate_rollout


class PPOTrain:

    def __init__(
        self,
        ppo_policy,
        ppo_reference,
        reward_model,
        value_model,
        policy_optimizer,
        value_optimizer,
        block_size,
        max_token,
        beta,
        epsilon,
        gamma,
        device,
        ppo_updates=2
    ):

        self.ppo_policy = ppo_policy
        self.ppo_reference = ppo_reference
        self.reward_model = reward_model
        self.value_model = value_model

        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer

        self.block_size = block_size
        self.max_token = max_token
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = torch.device(device)
        self.ppo_updates = ppo_updates

        if self.ppo_updates < 1:
            raise ValueError("ppo_updates must be >= 1")

        self.ppo_reference.requires_grad_(False)
        self.reward_model.requires_grad_(False)

        self.last_train_metrics = {}
        self.last_val_metrics = {}

    # -------------------------------------------------------
    # 1. Compute new policy log-probabilities
    # -------------------------------------------------------

    def compute_new_logprobs(self, full_ids, response_length):

        _, logits = self.ppo_policy(full_ids)

        logprobs = token_log_probs(logits, full_ids)

        return logprobs[:, -response_length:]

    # -------------------------------------------------------
    # 2. Get value predictions for response tokens
    # -------------------------------------------------------

    def compute_values(self, full_ids, prompt_length):

        values = self.value_model(full_ids)

        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)

        return values[:, prompt_length - 1:-1]

    # -------------------------------------------------------
    # 3. Generate rollout and compute fixed targets
    # -------------------------------------------------------

    def prepare_rollout(self, prompt_ids):

        P = prompt_ids.shape[1]

        with torch.no_grad():

            # Generate response and KL rewards
            (
                full_ids,
                old_logprobs,
                ref_logprobs,
                token_rewards
            ) = generate_rollout(
                self.ppo_policy,
                self.ppo_reference,
                prompt_ids,
                self.block_size,
                self.max_token,
                self.beta
            )

            response_length = full_ids.shape[1] - P

            if response_length <= 0:
                raise ValueError(
                    "Rollout generated no response tokens"
                )

            if full_ids.shape[1] > self.block_size:
                raise ValueError(
                    "Generated sequence exceeds block size"
                )

            expected_shape = (
                full_ids.shape[0],
                response_length
            )

            for name, tensor in [
                ("old_logprobs", old_logprobs),
                ("ref_logprobs", ref_logprobs),
                ("token_rewards", token_rewards)
            ]:

                if tensor.shape != expected_shape:
                    raise ValueError(
                        f"{name}: expected {expected_shape}, "
                        f"got {tuple(tensor.shape)}"
                    )

            # No padding in the rollout
            attention_mask = torch.ones_like(full_ids)

            # Reward for each completed response
            reward_scores = self.reward_model(
                full_ids,
                attention_mask
            ).reshape(-1)

            if reward_scores.numel() != full_ids.shape[0]:
                raise ValueError(
                    "Reward model must return one score "
                    "per generated sequence"
                )

            # Add terminal reward to token-level KL rewards
            rewards = token_rewards.clone()
            rewards[:, -1] += reward_scores

            # Discounted returns
            returns = cumulative_future_returns(
                rewards,
                gamma=self.gamma
            )

            # Old value predictions
            old_values = self.compute_values(
                full_ids,
                P
            )

            if old_values.shape != returns.shape:
                raise ValueError(
                    "Values and returns have different shapes"
                )

            # Fixed advantages
            advantages = returns - old_values

            # -------------------------------------------------------
            # Rollout statistics
            # -------------------------------------------------------

            log_ratio = old_logprobs - ref_logprobs

            rollout_stats = {
                "reward_sum": reward_scores.sum().item(),
                "kl_sum": log_ratio.sum().item(),
                "num_sequences": full_ids.shape[0],
                "num_tokens": log_ratio.numel()
            }

        return (
            full_ids,
            old_logprobs,
            returns,
            advantages,
            reward_scores,
            rollout_stats
        )

    # -------------------------------------------------------
    # 4. Compute PPO and critic losses
    # -------------------------------------------------------

    def compute_losses(
        self,
        full_ids,
        prompt_length,
        old_logprobs,
        returns,
        advantages
    ):

        response_length = old_logprobs.shape[1]

        new_logprobs = self.compute_new_logprobs(
            full_ids,
            response_length
        )

        # Policy loss
        policy_loss = PPO_Loss(
            new_logprobs,
            old_logprobs,
            advantages,
            self.epsilon
        )

        # Value loss
        values = self.compute_values(
            full_ids,
            prompt_length
        )

        if values.shape != returns.shape:
            raise ValueError(
                f"Value shape {values.shape} "
                f"does not match returns {returns.shape}"
            )

        value_loss = F.mse_loss(
            values,
            returns
        )

        total_loss = policy_loss + 0.5 * value_loss

        # -------------------------------------------------------
        # PPO clipping statistic
        # -------------------------------------------------------

        with torch.no_grad():

            ratio = torch.exp(
                new_logprobs.detach() - old_logprobs
            )

            clipped = (
                (ratio < 1 - self.epsilon)
                |
                (ratio > 1 + self.epsilon)
            )

            clip_fraction = clipped.float().mean().item()

        return (
            total_loss,
            policy_loss,
            value_loss,
            clip_fraction
        )

    # -------------------------------------------------------
    # 5. Run one training or validation epoch
    # -------------------------------------------------------

    def _run_epoch(self, data_loader, training):

        # Keep dropout disabled for stable PPO ratios.
        self.ppo_policy.eval()
        self.ppo_reference.eval()
        self.reward_model.eval()
        self.value_model.eval()

        # Total losses
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0

        # Total rollout results
        reward_sum = 0.0
        kl_sum = 0.0

        # Total ratio clipping
        clip_sum = 0.0

        num_sequences = 0
        num_tokens = 0
        loss_token_count = 0

        for batch_idx, prompt_ids in enumerate(data_loader):

            prompt_ids = prompt_ids.to(self.device)

            P = prompt_ids.shape[1]

            # -------------------------------------------------------
            # 1. Generate rollout
            # -------------------------------------------------------

            (
                full_ids,
                old_logprobs,
                returns,
                advantages,
                _,
                rollout_stats
            ) = self.prepare_rollout(prompt_ids)

            # Accumulate reward and KL statistics
            reward_sum += rollout_stats["reward_sum"]
            kl_sum += rollout_stats["kl_sum"]

            num_sequences += rollout_stats["num_sequences"]
            num_tokens += rollout_stats["num_tokens"]

            # -------------------------------------------------------
            # 2. PPO updates
            # -------------------------------------------------------

            num_updates = (
                self.ppo_updates if training else 1
            )

            for update in range(num_updates):
                (
                                    total_loss,
                                    policy_loss,
                                    value_loss,
                                    clip_fraction
                                ) = self.compute_losses(
                                    full_ids,
                                    P,
                                    old_logprobs,
                                    returns,
                                    advantages)

                # Number of response tokens in this batch
                n = rollout_stats["num_tokens"]

                # Accumulate metrics across update passes
                total_loss_sum += total_loss.item() * n
                policy_loss_sum += policy_loss.item() * n
                value_loss_sum += value_loss.item() * n

                clip_sum += clip_fraction * n
                loss_token_count += n

                if training:

                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()

                    total_loss.backward()

                    self.policy_optimizer.step()
                    self.value_optimizer.step()

        if num_sequences == 0:
            raise ValueError("PPO DataLoader is empty")

        # -------------------------------------------------------
        # 3. Compute epoch averages
        # -------------------------------------------------------

        metrics = {
            "total_loss": (
                total_loss_sum / max(loss_token_count, 1)
            ),
            "policy_loss": (
                policy_loss_sum / max(loss_token_count, 1)
            ),
            "value_loss": (
                value_loss_sum / max(loss_token_count, 1)
            ),
            "mean_reward": (
                reward_sum / num_sequences
            ),
            "mean_kl": (
                kl_sum / max(num_tokens, 1)
            ),
            "clip_fraction": (
                clip_sum / max(loss_token_count, 1)
            ),
            "response_length": (
                num_tokens / num_sequences
            )
        }

        if training:
            self.last_train_metrics = metrics
        else:
            self.last_val_metrics = metrics

        return metrics["total_loss"]

    # -------------------------------------------------------
    # 6. Train one epoch
    # -------------------------------------------------------

    def train_one_epoch(self, train_loader):

        return self._run_epoch(
            train_loader,
            training=True
        )

    # -------------------------------------------------------
    # 7. Evaluate
    # -------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, val_loader):

        return self._run_epoch(
            val_loader,
            training=False
        )

    # -------------------------------------------------------
    # 8. Full training
    # -------------------------------------------------------

    def full_training(
        self,
        train_loader,
        val_loader,
        num_epochs,
        eval_every=1
    ):

        history = {
            # Existing metrics
            "train_loss": [],
            "val_loss": [],
            "eval_epochs": [],

            # Training diagnostics
            "policy_loss": [],
            "value_loss": [],
            "mean_reward": [],
            "mean_kl": [],
            "clip_fraction": [],
            "response_length": [],

            # Validation diagnostics
            "val_policy_loss": [],
            "val_value_loss": [],
            "val_mean_reward": [],
            "val_mean_kl": [],
            "val_response_length": []
        }

        train_metric_names = [
            "policy_loss",
            "value_loss",
            "mean_reward",
            "mean_kl",
            "clip_fraction",
            "response_length"
        ]

        val_metric_names = [
            "policy_loss",
            "value_loss",
            "mean_reward",
            "mean_kl",
            "response_length"
        ]

        for epoch in range(1, num_epochs + 1):

            # -------------------------------------------------------
            # 1. Training
            # -------------------------------------------------------

            train_loss = self.train_one_epoch(
                train_loader
            )

            history["train_loss"].append(train_loss)

            for name in train_metric_names:

                history[name].append(
                    self.last_train_metrics[name]
                )

            # -------------------------------------------------------
            # 2. Validation
            # -------------------------------------------------------

            if (
                epoch % eval_every == 0
                or epoch == num_epochs
            ):

                val_loss = self.evaluate(
                    val_loader
                )

                history["val_loss"].append(val_loss)
                history["eval_epochs"].append(epoch)

                for name in val_metric_names:

                    history[f"val_{name}"].append(
                        self.last_val_metrics[name]
                    )

                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Reward: "
                    f"{self.last_train_metrics['mean_reward']:.4f} | "
                    f"KL: "
                    f"{self.last_train_metrics['mean_kl']:.4f} | "
                    f"Clip: "
                    f"{self.last_train_metrics['clip_fraction']:.4f}"
                )

            else:

                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Reward: "
                    f"{self.last_train_metrics['mean_reward']:.4f} | "
                    f"KL: "
                    f"{self.last_train_metrics['mean_kl']:.4f}"
                )

        return history
