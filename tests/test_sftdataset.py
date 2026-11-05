
#########################################################################
#                         Tests for SFTDataset
#########################################################################

import json
import torch
import pytest

from src.data.SFT_dataset import SFTDataset
from src.data.utils import get_prompt_response, get_full_text
from tinygpt.data import encode


# -------------------------------------------------------
# Step 1: Create a small artificial dataset
# -------------------------------------------------------

@pytest.fixture
def sample_data():
    return [
        {
            "instruction": "Say hi",
            "response": "Hello"
        },
        {
            "instruction": "Add 1 and 1",
            "response": "2"
        }
    ]


# -------------------------------------------------------
# Step 2:  Create a toy tokenizer
# -------------------------------------------------------

def build_vocab(data):

    # Include all characters introduced by formatting
    texts = [get_full_text(example, task = "SFT") for example in data]

    chars = sorted(set("".join(texts)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    return itos, stoi


# -------------------------------------------------------
# Step 3: Create the dataset
# -------------------------------------------------------

@pytest.fixture
def dataset(tmp_path, sample_data):

    path = tmp_path / "test_sft.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)

    itos, stoi = build_vocab(sample_data)

    return SFTDataset(
        path=path,
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

def test_shapes(dataset):

    x, y, loss_mask = dataset[0]

    assert x.shape == (128,)     # x.shape should be (block_size, )
    assert y.shape == (128,)     # y.shape should be (block_size, )
    assert loss_mask.shape == (128,)          # loss_mask.shape should be (block_size, )

    assert x.dtype == torch.long       # x.dtype should be torch.long 
    assert y.dtype == torch.long       # same for y 
    assert loss_mask.dtype == torch.long     # same for loss_mask


# -------------------------------------------------------
# Test 3: Next-token prediction alignment
# -------------------------------------------------------

def test_shift(dataset, sample_data):

    x, y, _ = dataset[0]

    full_text = get_full_text(sample_data[0], task = "SFT")
    tokens = encode(full_text, dataset.stoi)

    n = min(len(tokens) - 1, dataset.block_size)

    expected_x = torch.tensor(tokens[:n])  
    expected_y = torch.tensor(tokens[1:n+1])

    assert torch.equal(x[:n], expected_x)     # x is build as tokens[:n]
    assert torch.equal(y[:n], expected_y)     # y is build as tokens[1:n+1]


# -------------------------------------------------------
# Test 4: Loss masking
# -------------------------------------------------------

def test_loss_mask(dataset, sample_data):

    _, _, loss_mask = dataset[0]

    prompt, _ = get_prompt_response(sample_data[0], task = "SFT")
    full_text = get_full_text(sample_data[0],task = "SFT")

    prompt_len = len(encode(prompt, dataset.stoi))
    
    seq_len = min(
        len(encode(full_text, dataset.stoi)) - 1,
        dataset.block_size
    )

    expected_mask = torch.zeros(
        dataset.block_size,
        dtype=torch.long
    )

    # Supervise response targets only
    expected_mask[prompt_len - 1:seq_len] = 1

    assert torch.equal(loss_mask, expected_mask)

    # First response token must contribute to loss
    assert loss_mask[prompt_len - 1] == 1

    # Prompt targets must not contribute
    assert torch.all(loss_mask[:prompt_len - 1] == 0)

    # Padding must not contribute
    assert torch.all(loss_mask[seq_len:] == 0)


# -------------------------------------------------------
# Test 5: Padding
# -------------------------------------------------------

def test_padding(dataset, sample_data):

    x, y, loss_mask = dataset[0]

    full_text = get_full_text(sample_data[0], task = "SFT")
    seq_len = len(encode(full_text, dataset.stoi)) - 1

    assert torch.all(x[seq_len:] == 0)
    assert torch.all(y[seq_len:] == 0)
    assert torch.all(loss_mask[seq_len:] == 0)


# -------------------------------------------------------
# Test 6: Truncation
# -------------------------------------------------------

def test_truncation(tmp_path):

    data = [
        {
            "instruction": "Say hi",
            "response": "Hello" * 100
        }
    ]

    path = tmp_path / "long_sft.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    itos, stoi = build_vocab(data)

    prompt, _ = get_prompt_response(data[0], task = "SFT")
    prompt_len = len(encode(prompt, stoi))

    block_size = prompt_len + 2

    dataset = SFTDataset(
        path=path,
        block_size=block_size,
        itos=itos,
        stoi=stoi
    )

    x, y, loss_mask = dataset[0]

    assert x.shape == (block_size,)
    assert y.shape == (block_size,)
    assert loss_mask.shape == (block_size,)

    # No padding: the sequence was truncated
    assert loss_mask.sum().item() == 3


# -------------------------------------------------------
# Test 7: Prompt too long
# -------------------------------------------------------

def test_long_prompt(tmp_path):

    data = [
        {
            "instruction": "Say hi " * 100,
            "response": "Hello"
        }
    ]

    path = tmp_path / "long_prompt.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    itos, stoi = build_vocab(data)

    dataset = SFTDataset(
        path=path,
        block_size=16,
        itos=itos,
        stoi=stoi
    )

    with pytest.raises(ValueError, match="Prompt is too long"):
        dataset[0]
