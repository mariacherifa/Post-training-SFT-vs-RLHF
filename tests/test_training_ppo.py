##############################################################################
#                             Test PPO Training
##############################################################################

import torch
import pytest
import math

from torch.utils.data import DataLoader

from src.trainers.ppo_training import PPOTrain
from src.models.value_model import ValueModel
from src.models.reward_model import RewardModel
from src.models.reference_policies_rlhf import reference_policy
from tinygpt.model import TinyGPT


# -------------------------------------------------------
# 1. Create PPO trainer
# -------------------------------------------------------

@pytest.fixture
def ppo_train(tmp_path):

    torch.manual_seed(42)

    config = {
        "n_embd": 32,
        "vocab_size": 30,
        "num_heads": 4,
        "block_size": 16,
        "num_layers": 1,
        "dropout": 0.1
    }

    # Initialize SFT model
    sft_model = TinyGPT(**config)

    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "model_state_dict": sft_model.state_dict(),
            "config": config
        },
        checkpoint_path
    )

    # Initialize PPO and reference policies
    ppo_policy, ref_policy = reference_policy(
        config,
        checkpoint_path,
        device="cpu"
    )

    # Initialize reward model
    reward_model = RewardModel(
        config["n_embd"],
        config["num_heads"],
        config["block_size"],
        config["vocab_size"],
        config["dropout"],
        config["num_layers"]
    )

    # Initialize value model
    value_model = ValueModel(
        config["n_embd"],
        config["num_heads"],
        config["block_size"],
        config["vocab_size"],
        config["dropout"],
        config["num_layers"]
    )

    # Freeze reference policy
    ref_policy.requires_grad_(False)
    ref_policy.eval()

    # Freeze reward model
    reward_model.requires_grad_(False)
    reward_model.eval()

    # Initialize policy optimizer
    policy_optimizer = torch.optim.AdamW(
        ppo_policy.parameters(),
        lr=1e-3
    )

    # Initialize value optimizer
    value_optimizer = torch.optim.AdamW(
        value_model.parameters(),
        lr=1e-3
    )

    # PPO hyperparameters
    max_token = 3
    beta = 0.05
    epsilon = 0.2
    gamma = 1.0
    ppo_updates = 2

    # Initialize PPO trainer
    ppo_trainer = PPOTrain(
        ppo_policy,
        ref_policy,
        reward_model,
        value_model,
        policy_optimizer,
        value_optimizer,
        config["block_size"],
        max_token,
        beta,
        epsilon,
        gamma,
        device="cpu",
        ppo_updates=ppo_updates
    )

    return ppo_trainer


# -------------------------------------------------------
# 2. Create artificial training DataLoader
# -------------------------------------------------------

@pytest.fixture
def train_loader():

    prompt_ids = torch.tensor([
        [1, 2, 3, 4],
        [1, 4, 5, 6],
        [2, 3, 4, 5],
        [5, 6, 7, 8],
        [1, 2, 4, 6]
    ], dtype=torch.long)

    return DataLoader(
        prompt_ids,
        batch_size=2,
        shuffle=False
    )


# -------------------------------------------------------
# 3. Create validation DataLoader
# -------------------------------------------------------

@pytest.fixture
def val_loader():

    prompt_ids = torch.tensor([
        [1, 3, 5, 7],
        [2, 4, 6, 8]
    ], dtype=torch.long)

    return DataLoader(
        prompt_ids,
        batch_size=2,
        shuffle=False
    )


# -------------------------------------------------------
# 4. Test PPO initialization
# -------------------------------------------------------

def test_ppo_initialization(ppo_train):

    # PPO policy must be trainable
    for p in ppo_train.ppo_policy.parameters():
        assert p.requires_grad

    # Value model must be trainable
    for p in ppo_train.value_model.parameters():
        assert p.requires_grad

    # Reference policy must be frozen
    for p in ppo_train.ppo_reference.parameters():
        assert not p.requires_grad

    # Reward model must be frozen
    for p in ppo_train.reward_model.parameters():
        assert not p.requires_grad

    # Check number of PPO updates
    assert ppo_train.ppo_updates == 2

    # Check that optimizers update different models
    policy_ids = {
        id(p)
        for group in ppo_train.policy_optimizer.param_groups
        for p in group["params"]
    }

    value_ids = {
        id(p)
        for group in ppo_train.value_optimizer.param_groups
        for p in group["params"]
    }

    assert policy_ids.isdisjoint(value_ids)


# -------------------------------------------------------
# 5. Test rollout and advantage shapes
# -------------------------------------------------------

def test_prepare_rollout(ppo_train, train_loader):

    prompt_ids = next(iter(train_loader))

    P = prompt_ids.shape[1]

    (
        full_ids,
    old_logprobs,
        returns,
    advantages,
    reward_scores,
                    rollout_stats
    ) = ppo_train.prepare_rollout(prompt_ids)

    B = prompt_ids.shape[0]
    T = full_ids.shape[1] - P

    # Check generated sequence
    assert full_ids.shape == (B, P + T)

    assert 0 < T <= ppo_train.max_token

    # Check log-probabilities
    assert old_logprobs.shape == (B, T)

    # Check returns
    assert returns.shape == (B, T)

    # Check advantages
    assert advantages.shape == (B, T)

    # Reward model produces one reward per sequence
    assert reward_scores.shape == (B,)

    # Check numerical values
    assert torch.isfinite(old_logprobs).all()
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(reward_scores).all()

    # Fixed targets must not track gradients
    assert not old_logprobs.requires_grad
    assert not returns.requires_grad
    assert not advantages.requires_grad


# -------------------------------------------------------
# 6. Test terminal reward contribution
# -------------------------------------------------------

def test_terminal_reward(ppo_train, monkeypatch):

    # Create a deterministic rollout with zero KL rewards
    def fake_generate_rollout(
        policy,
        reference,
        prompt_ids,
        block_size,
        max_token,
        beta
    ):

        B = prompt_ids.shape[0]

        response_ids = torch.tensor(
            [9, 10, 11],
            dtype=torch.long,
            device=prompt_ids.device
        ).unsqueeze(0).expand(B, -1)

        full_ids = torch.cat(
            [prompt_ids, response_ids],
            dim=1
        )

        old_logprobs = torch.zeros(
            B, 3, device=prompt_ids.device
        )

        ref_logprobs = torch.zeros(
            B, 3, device=prompt_ids.device
        )

        token_rewards = torch.zeros(
            B, 3, device=prompt_ids.device
        )

        return (
            full_ids,
            old_logprobs,
            ref_logprobs,
            token_rewards
        )

    monkeypatch.setattr(
        "src.trainers.ppo_training.generate_rollout",
        fake_generate_rollout
    )

    prompt_ids = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5]
    ], dtype=torch.long)

    (
        full_ids,
                    old_logprobs,
                    returns,
                    advantages,
                    reward_scores,
                    rollout_stats
    ) = ppo_train.prepare_rollout(prompt_ids)

    # With gamma=1 and zero KL rewards:
    # each return must equal the terminal reward.
    expected_returns = reward_scores.unsqueeze(1).expand_as(
        returns
    )

    assert torch.allclose(
        returns,
        expected_returns,
        atol=1e-5
    )


# -------------------------------------------------------
# 7. Test one epoch training
# -------------------------------------------------------

def test_train_one_epoch(ppo_train, train_loader):

    # Save policy parameters before training
    old_policy_params = [
        p.detach().clone()
        for p in ppo_train.ppo_policy.parameters()
    ]

    # Save value model parameters
    old_value_params = [
        p.detach().clone()
        for p in ppo_train.value_model.parameters()
    ]

    # Save reference policy parameters
    old_ref_params = [
        p.detach().clone()
        for p in ppo_train.ppo_reference.parameters()
    ]

    # Save reward model parameters
    old_reward_params = [
        p.detach().clone()
        for p in ppo_train.reward_model.parameters()
    ]

    # Train one epoch
    total_avg_loss = ppo_train.train_one_epoch(
        train_loader
    )

    assert total_avg_loss is not None
    assert math.isfinite(total_avg_loss)

    # Check policy has updated
    policy_updated = any(
        not torch.equal(p_old, p_new)
        for p_old, p_new in zip(
            old_policy_params,
            ppo_train.ppo_policy.parameters()
        )
    )

    assert policy_updated

    # Check value model has updated
    value_updated = any(
        not torch.equal(p_old, p_new)
        for p_old, p_new in zip(
            old_value_params,
            ppo_train.value_model.parameters()
        )
    )

    assert value_updated

    # Reference policy must remain unchanged
    for p_old, p_new in zip(
        old_ref_params,
        ppo_train.ppo_reference.parameters()
    ):
        assert torch.equal(p_old, p_new)

    # Reward model must remain unchanged
    for p_old, p_new in zip(
        old_reward_params,
        ppo_train.reward_model.parameters()
    ):
        assert torch.equal(p_old, p_new)


# -------------------------------------------------------
# 8. Test evaluation
# -------------------------------------------------------

def test_evaluate(ppo_train, val_loader):

    # Save policy parameters
    old_policy_params = [
        p.detach().clone()
        for p in ppo_train.ppo_policy.parameters()
    ]

    # Save value model parameters
    old_value_params = [
        p.detach().clone()
        for p in ppo_train.value_model.parameters()
    ]

    # Evaluate
    val_loss = ppo_train.evaluate(val_loader)

    assert val_loss is not None
    assert math.isfinite(val_loss)

    # Policy must not change
    for p_old, p_new in zip(
        old_policy_params,
        ppo_train.ppo_policy.parameters()
    ):
        assert torch.equal(p_old, p_new)

    # Value model must not change
    for p_old, p_new in zip(
        old_value_params,
        ppo_train.value_model.parameters()
    ):
        assert torch.equal(p_old, p_new)


# -------------------------------------------------------
# 9. Test full training
# -------------------------------------------------------

def test_full_training(
    ppo_train,
    train_loader,
    val_loader
):

    num_epochs = 2
    eval_every = 1

    history = ppo_train.full_training(
        train_loader,
        val_loader,
        num_epochs,
        eval_every
    )

    # Check number of recorded losses
    assert len(history["train_loss"]) == num_epochs
    assert len(history["val_loss"]) == num_epochs

    # Check evaluation epochs
    assert history["eval_epochs"] == [1, 2]

    # Check training losses
    for loss in history["train_loss"]:
        assert math.isfinite(loss)

    # Check validation losses
    for loss in history["val_loss"]:
        assert math.isfinite(loss)



# -------------------------------------------------------
# Test PPO metrics history
# -------------------------------------------------------

def test_ppo_metrics_history(
    ppo_train,
    train_loader,
    val_loader
):

    history = ppo_train.full_training(
        train_loader,
        val_loader,
        num_epochs=2,
        eval_every=1
    )

    train_metrics = [
        "train_loss",
        "policy_loss",
        "value_loss",
        "mean_reward",
        "mean_kl",
        "clip_fraction",
        "response_length"
    ]

    val_metrics = [
        "val_loss",
        "val_policy_loss",
        "val_value_loss",
        "val_mean_reward",
        "val_mean_kl",
        "val_response_length"
    ]

    for metric in train_metrics:

        assert metric in history
        assert len(history[metric]) == 2

        assert all(
            math.isfinite(value)
            for value in history[metric]
        )

    for metric in val_metrics:

        assert metric in history
        assert len(history[metric]) == 2

        assert all(
            math.isfinite(value)
            for value in history[metric]
        )

    assert history["eval_epochs"] == [1, 2]

    for clip_fraction in history["clip_fraction"]:
        assert 0 <= clip_fraction <= 1

    for response_length in history["response_length"]:
        assert 0 < response_length <= ppo_train.max_token
