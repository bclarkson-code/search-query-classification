import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from transformers import GPT2Tokenizer
from glob import glob
from pathlib import Path
from datasets import Dataset, load_from_disk


class SearchQueryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "raw_datasets",
        batch_size: int = 128,
        num_workers: int = None,
        max_length: int = 24,
        debug=False,
        persistent_workers=True,
    ):

        super().__init__()
        self.data_path = os.path.join(os.getcwd(), data_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.persistent_workers = persistent_workers
        if num_workers is None:
            self.num_workers = os.cpu_count()
        else:
            self.num_workers = num_workers
        self.tokeniser = GPT2Tokenizer.from_pretrained(
            "gpt2",
            max_len=self.max_length,
            truncation=True,
            padding="max_length",
        )
        if os.path.exists(f"datasets/train"):
            self.train = load_from_disk(f"datasets/train")
            self.train.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "category"],
            )
        else:
            self.train = None
        if os.path.exists(f"datasets/valid"):
            self.valid = load_from_disk(f"datasets/valid")
            self.valid.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "category"],
            )
        else:
            self.valid = None
        if os.path.exists(f"datasets/test"):
            self.test = load_from_disk(f"datasets/test")
            self.test.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "category"],
            )
        else:
            self.test = None

    def shell_command(self, cmd):
        cmd = shlex.split(cmd)
        with Popen(cmd, stdout=PIPE) as proc:
            for line in proc.stdout:
                print(line.decode("utf-8"), end="")

    def prepare_dataset(self):
        shell_command("bash prepare_dataset.sh")
        return dataset

    def prepare_data(self):
        if not os.path.exists(f"datasets/train"):
            self.train = self.prepare_dataset()
        if not os.path.exists(f"datasets/valid"):
            self.valid = self.prepare_dataset()
        if not os.path.exists(f"datasets/test"):
            self.test = self.prepare_dataset()

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


if __name__ == "__main__":
    ds = SearchQueryDataModule()
    ds.prepare_data()
