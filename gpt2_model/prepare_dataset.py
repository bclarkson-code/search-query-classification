import os
from datasets import Dataset
from transformers import GPT2Tokenizer
import pandas as pd


def get_encoding(keys, encoding):
    if type(keys) == str:
        keys = [keys]
    return [encoding[key] for key in keys]


def encode_ds(ds, encoding):
    return ds.map(
        lambda x: dict(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            category=get_encoding(x["category"], encoding=encoding),
        ),
        batched=True,
    )


if __name__ == "__main__":
    categories = [
        "Computers",
        "Regional",
        "Home",
        "World",
        "Reference",
        "Society",
        "Health",
        "News",
        "Business",
        "Arts",
        "Science",
        "Recreation",
        "Shopping",
        "Sports",
        "Games",
    ]
    encoding = {cat: i for i, cat in enumerate(categories)}
    tokeniser = GPT2Tokenizer.from_pretrained(
        "gpt2", max_len=24, truncation=True, padding="max_length"
    )
    tokeniser.pad_token = tokeniser.eos_token

    for dataset_path in ["train", "test", "valid"]:
        # Load dataset
        print("Reading pickle")
        ds_file = f"datasets/{dataset_path}.pkl"
        df = pd.read_pickle(ds_file)
        # Clean dataset
        print("Cleaning")
        df = df[~pd.isna(df["category"])]
        df = df[~pd.isna(df["query"])]
        df = df[["query", "category"]]

        # Convert from pandas into huggingface dataset
        print("Converting to huggingface dataset")
        dataset = Dataset.from_pandas(df)

        # Tokenise queries
        print("Tokenising")
        dataset = dataset.map(
            lambda ex: tokeniser(
                ex["query"],
                add_special_tokens=True,
                truncation=True,
                max_length=24,
                padding="max_length",
            ),
            batched=True,
        )

        # Numerically encode categories
        print("Numerically Encoding categories")
        dataset = encode_ds(dataset, encoding)
        print("Saving dataset")
        dataset.save_to_disk(f"datasets/{dataset_path}")
