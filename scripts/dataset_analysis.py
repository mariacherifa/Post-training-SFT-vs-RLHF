
######################################################################
#                    SHP Dataset Analysis
######################################################################

import torch
import json
import argparse

from pathlib import Path
from collections import Counter
from itertools import islice
from datasets import load_dataset

from src.data.utils import get_full_text


# ====================================================================
# 1. Convert SHP dataset
# ====================================================================

def convert_shp(example):

    instruction = example["history"].strip()

    if example["labels"] == 1:
        accepted = example["human_ref_A"]
        rejected = example["human_ref_B"]

    elif example["labels"] == 0:
        accepted = example["human_ref_B"]
        rejected = example["human_ref_A"]

    else:
        raise ValueError("Invalid preference label")

    accepted = accepted.strip()
    rejected = rejected.strip()

    if not instruction or not accepted or not rejected:
        return None

    if accepted == rejected:
        return None

    return {
        "instruction": instruction,
        "accepted": accepted,
        "rejected": rejected,
        "post_id": example["post_id"],
        "domain": example["domain"],
        "score_ratio": example["score_ratio"]
    }


# ====================================================================
# 2. Format text for SFT
# ====================================================================

def formatted_text(example, response_key):

    data = {
        "instruction": example["instruction"],
        "response": example[response_key]
    }

    return get_full_text(data, task="SFT")


# ====================================================================
# 3. Compute percentile
# ====================================================================

def percentile(values, p):

    values = sorted(values)

    index = round((len(values) - 1) * p)

    return values[index]


# ====================================================================
# 4. Analyze dataset
# ====================================================================

def analyze_dataset(data, block_size, stoi=None):

    if len(data) == 0:
        print("Empty dataset")
        return

    n = len(data)

    prompt_lengths = []
    accepted_lengths = []
    rejected_lengths = []

    fit_context = 0
    fit_vocab = 0
    fit_both = 0

    domains = []

    for example in data:

        instruction = example["instruction"]
        accepted = example["accepted"]
        rejected = example["rejected"]

        # Lengths
        prompt_lengths.append(len(instruction))
        accepted_lengths.append(len(accepted))
        rejected_lengths.append(len(rejected))

        domains.append(example["domain"])

        # Complete formatted sequences
        full_accepted = formatted_text(example, "accepted")
        full_rejected = formatted_text(example, "rejected")

        # Check context size
        context_ok = (
            len(full_accepted) - 1 <= block_size
            and len(full_rejected) - 1 <= block_size
        )

        if context_ok:
            fit_context += 1

        # Check vocabulary
        if stoi is not None:

            vocab_ok = True

            for char in full_accepted + full_rejected:

                if char not in stoi:
                    vocab_ok = False
                    break

            if vocab_ok:
                fit_vocab += 1

            if vocab_ok and context_ok:
                fit_both += 1

    # Print results
    print(f"Number of examples: {n}")

    for name, lengths in [
        ("Instruction", prompt_lengths),
        ("Accepted", accepted_lengths),
        ("Rejected", rejected_lengths)
    ]:

        median = percentile(lengths, 0.5)
        p90 = percentile(lengths, 0.9)
        maximum = max(lengths)

        print(
            f"{name}: "
            f"median={median}, "
            f"p90={p90}, "
            f"max={maximum}"
        )

    print(f"Fit context: {100 * fit_context / n:.2f}%")

    if stoi is not None:

        print(f"Fit vocabulary: {100 * fit_vocab / n:.2f}%")
        print(f"Fit both: {100 * fit_both / n:.2f}%")

    # Domains distribution
    print("\nDomains:")

    domain_counts = Counter(domains)

    for domain, count in domain_counts.most_common():
        print(f"{domain}: {count}")


# ====================================================================
# 5. Analyze different context sizes
# ====================================================================

def analyze_context_sizes(data):

    block_sizes = [128, 256, 512, 1024, 2048]

    if len(data) == 0:
        print("Empty dataset")
        return

    n = len(data)

    print("\nContext Window Analysis")

    for block_size in block_sizes:

        n_compatible = 0

        for example in data:

            full_accepted = formatted_text(example, "accepted")
            full_rejected = formatted_text(example, "rejected")

            len_accepted = len(full_accepted) - 1
            len_rejected = len(full_rejected) - 1

            if (
                len_accepted <= block_size
                and len_rejected <= block_size
            ):
                n_compatible += 1

        percentage = 100 * n_compatible / n

        print(
            f"block_size={block_size}: "
            f"{n_compatible}/{n} "
            f"({percentage:.2f}%)"
        )


# ====================================================================
# 6. Save dataset
# ====================================================================

def save_json(data, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ====================================================================
# 7. Main function
# ====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--block-size",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Load TinyGPT vocabulary
    # -------------------------------------------------------

    stoi = None

    if args.checkpoint is not None:

        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True
        )

        stoi = checkpoint["stoi"]

    # -------------------------------------------------------
    # Dataset configuration
    # -------------------------------------------------------

    output_dir = Path("data/shp_processed")

    split_limits = {
        "train": 3000,
        "validation": 500,
        "test": 500
    }

    # -------------------------------------------------------
    # Load and process SHP
    # -------------------------------------------------------

    for split, limit in split_limits.items():

        print("\n" + "=" * 50)
        print(f"Processing {split}")
        print("=" * 50)

        dataset = load_dataset(
            "stanfordnlp/SHP",
            split=split,
            streaming=True
        )

        dataset = dataset.shuffle(
            seed=42,
            buffer_size=10000
        )

        processed = []

        for example in islice(dataset, limit):

            converted = convert_shp(example)

            if converted is not None:
                processed.append(converted)

        # ---------------------------------------------------
        # Analyze dataset
        # ---------------------------------------------------

        print("\nDataset Statistics")

        analyze_dataset(
            processed,
            args.block_size,
            stoi
        )

        # ---------------------------------------------------
        # Analyze context sizes
        # ---------------------------------------------------

        analyze_context_sizes(processed)

        # ---------------------------------------------------
        # Save processed dataset
        # ---------------------------------------------------

        path = output_dir / f"{split}_preferences.json"

        save_json(processed, path)

        print(f"\nSaved {len(processed)} examples.")

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
