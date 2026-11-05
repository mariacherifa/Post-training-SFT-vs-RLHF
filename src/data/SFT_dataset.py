#########################################################################
#                           SFT Dataset
#########################################################################

import torch
from torch.utils.data import Dataset

from .utils import load_data, get_prompt_response, get_full_text
from tinygpt.data import encode


class SFTDataset(Dataset):

    def __init__(self, path, block_size, itos, stoi):
        super().__init__()

        if block_size < 1:
            raise ValueError("block_size must be positive.")

        self.block_size = block_size
        self.itos = itos
        self.stoi = stoi

        # Load instruction-response examples 
        self.examples = load_data(path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        # Get a complete example instruction and response

        example = self.examples[index]

        # Seperate prompt and response 

        prompt, response = get_prompt_response(example, task = "SFT")
        full_text = get_full_text(example, task = "SFT")

        # Ensure correct alignment for loss masking

        if full_text != prompt + response:
            raise ValueError(
                "full_text must equal prompt + response."
            )

        # Encode the text with pretrained tokenizer

        prompt_encoded = encode(prompt, self.stoi)
        full_text_encoded = encode(full_text, self.stoi)

        prompt_len = len(prompt_encoded)

        if prompt_len == 0:
            raise ValueError("Prompt cannot be empty.")

        if len(full_text_encoded) <= prompt_len:
            raise ValueError("Response cannot be empty.")

        # Check prompt length

        if prompt_len > self.block_size:
            raise ValueError(
                "Prompt is too long for the block_size."
            )

       
        # Keep block_size + 1 tokens because x and y will be shifted relative to each other.

        tokens = full_text_encoded[:self.block_size + 1]

        # Construct x and y

        x = tokens[:-1]
        y = tokens[1:]

        seq_len = len(y)

        #  Construct loss mask : Ignore prompt, start supervising at the first response token.

        loss_mask = [0] * (prompt_len - 1)

        loss_mask += [1] * (seq_len - len(loss_mask))

        # Padding

        pad_len = self.block_size - seq_len

        if pad_len > 0:
            x += [0] * pad_len
            y += [0] * pad_len
            loss_mask += [0] * pad_len

        # Convert to tensors

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        loss_mask = torch.tensor(
            loss_mask, dtype=torch.long
        )

        return x, y, loss_mask
