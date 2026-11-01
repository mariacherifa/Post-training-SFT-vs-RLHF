
##################################################
#                Preference Dataset
##################################################

from torch.utils.data import Dataset
from .utils import load_data, get_prompt_response
from tinygpt.data import encode
import torch


class PreferenceDataset(Dataset):

    def __init__(self, path_file, block_size, itos, stoi):
        super().__init__()

        if block_size < 1:
            raise ValueError("block_size must be positive.")

        self.block_size = block_size
        self.examples = load_data(path_file)
        self.itos = itos
        self.stoi = stoi

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        prompt, rejected, accepted = get_prompt_response(
            example, task="RL"
        )

        # Check nonempty fields

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if not rejected:
            raise ValueError("Rejected answer cannot be empty.")

        if not accepted:
            raise ValueError("Accepted answer cannot be empty.")

        # Construct complete sequences
        chosen_text = prompt + accepted
        rejected_text = prompt + rejected

        # Encode
        chosen_encoded = encode(chosen_text, self.stoi)
        rejected_encoded = encode(rejected_text, self.stoi)

        # Check lengths
        if len(chosen_encoded) > self.block_size:
            raise ValueError("Chosen sequence exceeds block_size.")

        if len(rejected_encoded) > self.block_size:
            raise ValueError("Rejected sequence exceeds block_size.")

        # Padding lengths
        pad_len_cho = self.block_size - len(chosen_encoded)
        pad_len_rej = self.block_size - len(rejected_encoded)

        # Pad sequences
        chosen_ids = chosen_encoded + [0] * pad_len_cho
        rejected_ids = rejected_encoded + [0] * pad_len_rej

        # Valid-token masks to ignore padding
        chosen_mask = (
            [1] * len(chosen_encoded) + [0] * pad_len_cho
        )

        rejected_mask = (
            [1] * len(rejected_encoded) + [0] * pad_len_rej
        )

        return (
            torch.tensor(chosen_ids, dtype=torch.long),
            torch.tensor(rejected_ids, dtype=torch.long),
            torch.tensor(chosen_mask, dtype=torch.bool),
            torch.tensor(rejected_mask, dtype=torch.bool)
        )
