#####################################################################
#                        Prepare experiment 
#####################################################################

import torch 
from tinygpt.data import build_vocab
import json 
from src.data.utils import get_full_text
from pathlib import Path


# Load json files 

def load_json(path_file): 

    with open(path_file, "r", encoding="utf-8") as f: 
        text = json.load(f)

    return text 

def load_checkpoint_vocab(checkpoint_path, device): 

    checkpoints = torch.load(checkpoint_path,
                             map_location= device,
                             weights_only=True)
    
    itos = checkpoints["itos"]
    stoi = checkpoints["stoi"]
    block_size = checkpoints["config"]["block_size"]

    return itos, stoi, block_size

def filter_compatible_examples(preference_data, stoi, block_size, max_new_token):

    filtered_data = []

    for example in preference_data: 

        instruction = example["instruction"]
        accepted = example["accepted"]
        rejected = example["rejected"]

        full_accepted = get_full_text(
            {"instruction": instruction, "response": accepted}, task="SFT"
        )
        full_rejected = get_full_text(
            {"instruction": instruction, "response": rejected}, task="SFT"
        )
        prompt = get_full_text(
            {"instruction": instruction, "response": ""}, task="SFT"
        )

        vocab_ok = all(char in stoi for char in full_accepted + full_rejected)
        context_ok_acc = 1 < len(full_accepted) <= block_size
        context_ok_rejected = 1 < len(full_rejected) <= block_size
        context_ok_prompt = 1 <= len(prompt) and len(prompt) + max_new_token <= block_size

        if vocab_ok and context_ok_acc and context_ok_rejected and context_ok_prompt: 

            filtered_data.append(example)

    return filtered_data

def create_training_views(filtered_data): 

    sft_data = []
    reward_data = []
    ppo_data = []

    post_seen = set()

    for example in filtered_data: 
        instruction = example["instruction"]
        accepted = example["accepted"]
        rejected = example["rejected"]
        post_id = example.get("post_id", instruction)

        sft_data.append({"instruction" : instruction,
                         "response" : accepted})

        reward_data.append({"instruction" : instruction,
                            "accepted" : accepted, 
                            "rejected": rejected
                            })

        if post_id not in post_seen: 
            ppo_data.append({"instruction": instruction})
            post_seen.add(post_id)


    return sft_data, reward_data, ppo_data

# Saving experiment data 
def save_experiment_data(data, output_path):

    output_path = Path(output_path)

    # Create the directory if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save data as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} examples to {output_path}")



        


    
     



