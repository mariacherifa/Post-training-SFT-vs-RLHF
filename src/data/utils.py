################################################################################
#                          Functions to handle data 
################################################################################
import json 
import torch.nn as nn

def load_data(path_file): 
    with open(path_file, "r", encoding= "utf-8") as f: 
        text = json.load(f)

    return text 

def get_prompt_response(example, task):

    if task == "SFT": 
        return example["instruction"], example["response"]
    if task == "RL":
        return example["instruction"], example["rejected"],example["accepted"]
 


def get_full_text(example, task):

    if task == "SFT": 
        full_text = example["instruction"] + example["response"]
        return full_text
    if task == "RL": 
        full_text = example["instruction"] + example["accepted"] + example["rejected"]
        return full_text




#####################################################################
#                     Token log probs  for RLHF
#####################################################################

# log probability of y_t knowing the context y_1,...y_{t-1}

def token_log_probs(logits, input_ids):

    # logits: (B, T, V)
    # input_ids: (B, T)

    # Align predictions with next-token targets
    x = logits[:, :-1, :]
    target = input_ids[:, 1:]

    # Log-probabilities over vocabulary
    log_probs = nn.functional.log_softmax(x, dim=-1)

    # Select the log-probability of each target token
    log_probs_target = log_probs.gather(
        dim=-1,
        index=target.unsqueeze(-1)
    ).squeeze(-1)

    return log_probs_target  # (B, T-1)

# KL penality in the RLHF objective 

def kl_penality(old_log_probs, ref_log_probs,response_mask, beta):

    if old_log_probs.shape != ref_log_probs.shape: 
        raise ValueError("probs need to have same shape")
    
    if old_log_probs.shape != response_mask.shape: 
        raise ValueError("mask and probs need to have same shape")


    kl_penality = -beta *(old_log_probs - ref_log_probs)

    kl_penality = kl_penality * response_mask

    return kl_penality


