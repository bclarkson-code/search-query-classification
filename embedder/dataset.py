import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import Dataset
import torch


class Subset(torch.utils.data.Dataset):
    def __init__(self, ds, label):
        self.ds = ds
        self.label = label

    def __len__(self):
        return 4096

    def __getitem__(self, idx):
        input = self.ds.__getitem__(idx)
        input["label"] = self.label
        return input


class LabelledDataset(torch.utils.data.Dataset):
    def __init__(self, ds, label):
        self.ds = ds
        self.label = label

    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, idx):
        input = self.ds.__getitem__(idx)
        input["label"] = self.label
        return input


class EmbedderData(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str = "dataframes",
        batch_size: int = 1024,
        num_workers: int = 1,
        persistent_workers: bool = True,
    ):

        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        train_path = os.path.join(data_path, "train")
        if os.path.exists(train_path):
            self.train = Dataset.load_from_disk(train_path)
            self.train.set_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
            )
        else:
            self.train = None
        valid_path = os.path.join(data_path, "valid")
        if os.path.exists(valid_path):
            self.valid = Dataset.load_from_disk(valid_path)
            self.valid.set_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
            )
        else:
            self.valid = None
        test_path = os.path.join(data_path, "test")
        if os.path.exists(test_path):
            self.test = Dataset.load_from_disk(test_path)
            self.test.set_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
            )
        else:
            self.test = None

    def train_dataloader(self):
        return DataLoader(
            LabelledDataset(self.train, "train"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            LabelledDataset(self.valid, "valid"),
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            LabelledDataset(self.test, "test"),
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def debug_dataloader(self):
        subset = Subset(self.train, "debug")
        return DataLoader(
            subset,
            batch_size=16,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )
