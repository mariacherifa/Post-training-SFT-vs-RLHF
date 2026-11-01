##########################################################################
#                       sampling the next token for RLHF
##########################################################################

from tinygpt.model import TinyGPT
import torch.nn as nn
import torch
from data.utils import kl_penality, token_log_probs
from models.reward_model import RewardModel

def generate_next_token(ppo_policy, encoded_prompt, block_size, max_token):

    ppo_policy.eval()

    old_logprobs = []

    with torch.no_grad():

        for _ in range(max_token):

            # Respect the context window
            idx_cond = encoded_prompt[:, -block_size:]

            # Get model logits
            logits, _ = ppo_policy(idx_cond)

            # Next-token logits
            logits = logits[:, -1, :]  # (B, V)

            # Compute probabilities
            probs = nn.functional.softmax(logits, dim=-1)

            # Compute log probabilities
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            # Sample next token
            idx = torch.multinomial(
                probs, num_samples=1
            )  # (B, 1)

            # Log probability of sampled token
            selected_logprob = log_probs.gather(
                dim=-1,
                index=idx
            )  # (B, 1)

            # Store sampled token log probability
            old_logprobs.append(selected_logprob)

            # Append generated token
            encoded_prompt = torch.cat(
                (encoded_prompt, idx), dim=-1
            )

    # Combine log probabilities
    old_logprobs = torch.cat(
        old_logprobs, dim=1
    )  # (B, max_token)

    return encoded_prompt, old_logprobs


# Now we need rollout generation 
def generate_rollout(ppo_policy,ppo_reference, encoded_prompt, block_size, max_token, beta): 

    full_ids, old_logprobs = generate_next_token(ppo_policy, encoded_prompt, block_size, max_token)

    if len(full_ids) > block_size: 
        raise ValueError("prompt + response too long ")

    ppo_reference.eval()

    with torch.no_grad():

        logits, _  = ppo_reference(full_ids)

        ref_logprobs =  token_log_probs(logits, full_ids)

        ref_logprobs = ref_logprobs[:, -max_token:]

    response_mask = torch.ones_like(old_logprobs)
    kl_pena = kl_penality(old_logprobs, ref_logprobs,response_mask, beta)

    return full_ids, old_logprobs, ref_logprobs, kl_pena

def rewards_sampled_trajectories(ppo_policy,
    ppo_reference,
    reward_model,
    encoded_prompt,
    block_size,
    max_token,
    beta): 

    full_ids, old_logprobs, ref_logprobs, kl_pena = generate_rollout(ppo_policy,ppo_reference, 
                                                                     encoded_prompt, 
                                                                     block_size, 
                                                                     max_token, 
                                                                     beta)


    attention_mask = torch.ones_like(full_ids, dtype=torch.bool)
    reward = reward_model(full_ids,attention_mask)
    token_rewards = kl_pena.clone()
    token_rewards[:, -1] += reward


    return full_ids, old_logprobs,ref_logprobs, token_rewards




    





























