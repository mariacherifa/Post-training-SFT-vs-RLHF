
##########################################################################
#                       Sampling the next token for RLHF
##########################################################################

import torch
import torch.nn.functional as F

from src.data.utils import kl_penality, token_log_probs


# -------------------------------------------------------
# 1. Generate response tokens
# -------------------------------------------------------

def generate_next_token(
    ppo_policy,
    encoded_prompt,
    block_size,
    max_token
):

    ppo_policy.eval()

    if encoded_prompt.ndim != 2:
        raise ValueError("Prompt must have shape (B, P)")

    if encoded_prompt.shape[1] == 0:
        raise ValueError("Prompt cannot be empty")

    if max_token <= 0:
        raise ValueError("max_token must be positive")

    # Check context length before generation

    if encoded_prompt.shape[1] + max_token > block_size:
        raise ValueError(
            "Prompt + generated response exceeds block size"
        )

    old_logprobs = []

    with torch.no_grad():

        for _ in range(max_token):

            # Respect context window
            idx_cond = encoded_prompt[:, -block_size:]

            # Get policy logits
            _, logits = ppo_policy(idx_cond)

            # Next-token logits
            logits = logits[:, -1, :]  # (B, V)

            # Compute log-probabilities
            log_probs = F.log_softmax(
                logits,
                dim=-1
            )

            # Sample next token
            probs = log_probs.exp()

            idx = torch.multinomial(
                probs,
                num_samples=1
            )  # (B, 1)

            # Log probability of sampled token
            selected_logprob = log_probs.gather(
                dim=-1,
                index=idx
            )  # (B, 1)

            old_logprobs.append(
                selected_logprob
            )

            # Append generated token
            encoded_prompt = torch.cat(
                (encoded_prompt, idx),
                dim=-1
            )

    # Combine log-probabilities
    old_logprobs = torch.cat(
        old_logprobs,
        dim=1
    )  # (B, max_token)

    return encoded_prompt, old_logprobs


# -------------------------------------------------------
# 2. Generate rollout and KL penalties
# -------------------------------------------------------

def generate_rollout(
    ppo_policy,
    ppo_reference,
    encoded_prompt,
    block_size,
    max_token,
    beta
):

    # Generate response using current PPO policy
    full_ids, old_logprobs = generate_next_token(
        ppo_policy,
        encoded_prompt,
        block_size,
        max_token
    )

    # Check context
    if full_ids.shape[1] > block_size:
        raise ValueError(
            "Prompt + response too long"
        )

    ppo_reference.eval()

    with torch.no_grad():

        # Compute reference log-probabilities
        _, logits = ppo_reference(full_ids)

        ref_logprobs = token_log_probs(
            logits,
            full_ids
        )

        # Keep response-token log-probabilities
        ref_logprobs = ref_logprobs[:, -max_token:]

    # Response mask we can use both: old_logprobs or ref_log_probs
    response_mask = torch.ones_like(
        old_logprobs
    )

    # Compute KL penalty
    kl_pena = kl_penality(
        old_logprobs,
        ref_logprobs,
        response_mask,
        beta
    )

    return (
        full_ids,
        old_logprobs,
        ref_logprobs,
        kl_pena
    )


# -------------------------------------------------------
# 3. Compute reward for sampled trajectories
# -------------------------------------------------------

def rewards_sampled_trajectories(
    ppo_policy,
    ppo_reference,
    reward_model,
    encoded_prompt,
    block_size,
    max_token,
    beta
):

    (
        full_ids,
        old_logprobs,
        ref_logprobs,
        kl_pena
    ) = generate_rollout(
        ppo_policy,
        ppo_reference,
        encoded_prompt,
        block_size,
        max_token,
        beta
    )

    reward_model.eval()

    with torch.no_grad():

        # No padding in generated sequence (attention mask is just equal to 1)
        attention_mask = torch.ones_like(
            full_ids,
            dtype=torch.long
        )

        # Reward for entire generated response
        reward = reward_model(
            full_ids,
            attention_mask
        ).reshape(-1)

        # Token-level rewards cloned from KL computed for each token.
        token_rewards = kl_pena.clone()

        # Add sequence reward at final response token
        token_rewards[:, -1] += reward

    return (
        full_ids,
        old_logprobs,
        ref_logprobs,
        token_rewards
    )
