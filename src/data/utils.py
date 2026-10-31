################################################################################
#                          Utils functions to handle data 
################################################################################
import json 

def load_data(path_file): 
    with open(path_file, "r", encoding= "utf-8") as f: 
        text = json.load(f)

    return text 

def get_prompt_response(example, task):

    if task == "SFT": 
        return example["instruction"], example["response"]
    if task == "RL":
        return example["instruction"], example["rejected"],example["accepted"]
 


def get_full_text(example, task ):

    if task == "SFT": 
        full_text = example["instruction"] + example["response"]
        return full_text
    if task == "RL": 
        full_text = example["instruction"] + example["rejected"] + example["accepted"]
        return full_text




