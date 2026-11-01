##############################################################################
#                             Test preference dataset
##############################################################################
import torch 
import pytest
import json
from src.data.utils import get_prompt_response, get_full_text
from src.data.PreferenceDataSet import PreferenceDataset
from tinygpt.data import encode

# -------------------------------------------------------
# 1. Create a small artificial dataset
# -------------------------------------------------------

@pytest.fixture

def sample_data(): 
    return  [{ "instruction" : "what is capital of France?", 

             "accepted" : "Paris", 

             "rejected" : "New york"
             }, 
             { "instruction" : "Who is the president of France?", 

             "accepted" : "Emmanuel Macron", 

             "rejected" : "Donald Trump"
             }
    ]

# -------------------------------------------------------
# 2. Create a toy vocabulary
# -------------------------------------------------------

def create_vocab(data): 

    texts = [get_full_text(example, task = "RL") for example in data]

    chars = sorted(set("".join(texts)))

    itos = { i : ch for i,ch in enumerate(chars)}
    stoi = {ch : i for i, ch in enumerate(chars)}

    return itos, stoi


# -------------------------------------------------------
# 2. Create dataset
# -------------------------------------------------------

@pytest.fixture
def dataset(tmp_path, sample_data):

    path = tmp_path / "test_sft.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)

    itos, stoi = create_vocab(sample_data)

    return PreferenceDataset(
        path_file=path,
        block_size=128,
        itos=itos,
        stoi=stoi
    )

# -------------------------------------------------------
# Test 1: Dataset length
# -------------------------------------------------------

def test_length(dataset): 
    assert len(dataset) == 2


# -------------------------------------------------------
# Test 2: Shapes and data types
# -------------------------------------------------------

def test_shapes_types(dataset): 

    chosen_ids, rejected_ids, chosen_mask, rejected_mask = dataset[0]

    assert chosen_ids.shape == (128,)
    assert rejected_ids.shape == (128,)
    assert chosen_mask.shape == (128,)
    assert rejected_mask.shape == (128,)

    assert chosen_ids.dtype == torch.long
    assert rejected_ids.dtype == torch.long
    assert chosen_mask.dtype == torch.bool
    assert rejected_mask.dtype == torch.bool


# -------------------------------------------------------
# Test 3: Next-token prediction alignment/ masking
# -------------------------------------------------------

def test_alignement_next_token(dataset, sample_data): 

    chosen_ids, rejected_ids, chosen_mask, rejected_mask = dataset[0]

    

    prompt, rejected, chosen = get_prompt_response(sample_data[0], task="RL")

    

    tokens_cho = encode(prompt + chosen, dataset.stoi)
    tokens_rej = encode(prompt + rejected, dataset.stoi)

    pad_len_cho = dataset.block_size - len(tokens_cho)
    pad_len_rej = dataset.block_size - len(tokens_rej)

    tokens_chosen = tokens_cho + [0] * pad_len_cho
    tokens_rejected = tokens_rej + [0] * pad_len_rej

    tokens_chosen  = torch.tensor(tokens_chosen, dtype= torch.long)
    tokens_rejected = torch.tensor(tokens_rejected, dtype= torch.long)


    assert torch.equal(chosen_ids, tokens_chosen)
    assert torch.equal(rejected_ids, tokens_rejected)
    assert torch.equal(chosen_mask, torch.tensor(([1]* len(tokens_cho) + [0]* (pad_len_cho)), dtype=torch.bool))
    assert torch.equal(rejected_mask ,torch.tensor(([1]* len(tokens_rej) + [0]* (pad_len_rej)), dtype=torch.bool))

    
# -------------------------------------------------------
# Test 3: padding
# -------------------------------------------------------

def test_padding(dataset, sample_data): 

    chosen_ids, rejected_ids, _, _ = dataset[0]
    
    prompt, rejected, chosen = get_prompt_response(sample_data[0], task="RL")

    token_cho = encode(prompt + chosen, dataset.stoi)
    token_rej = encode(prompt + rejected, dataset.stoi)

    pad_chosen_len = dataset.block_size - len(token_cho)
    pad_rejected_len = dataset.block_size - len(token_rej)

    pad_chosen = torch.tensor([0]*pad_chosen_len, dtype= torch.long )
    pad_rejected = torch.tensor([0]*pad_rejected_len, dtype= torch.long )

    assert torch.equal(chosen_ids[len(token_cho):], pad_chosen)
    assert torch.equal(rejected_ids[len(token_rej):], pad_rejected)


# -------------------------------------------------------
# Test 7: Prompt too long
# -------------------------------------------------------

def test_long_prompt(tmp_path):

    data = [
        {
            "instruction": "Say hi " * 100,
            "accepted": "Hello", 
            "rejected": "No"
        }
    ]

    path = tmp_path / "long_prompt.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    itos, stoi = create_vocab(data)

    dataset =PreferenceDataset(
        path_file=path,
        block_size=16,
        itos=itos,
        stoi=stoi
    )



    with pytest.raises(ValueError, match="Chosen sequence exceeds block_size"):
            dataset[0]
    
