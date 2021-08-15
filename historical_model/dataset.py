import torch
import numpy as np
from pytorch_lightning import LightningDataModule
import pandas as pd
from torch.utils.data import DataLoader
import pickle5 as pickle

class HistoricalModelDataset(torch.utils.data.Dataset):
    n_labels = 16
    def __init__(self, df, encoding):
        self.encoding = encoding
        print(encoding)
        raise ValueError
        self.hist_embedding = df['historical_embedding']
        self.hist_label = df['historical_label']
        self.query_embedding = df[range(768)]
        self.category = df['category'].replace(encoding)

    def __getitem__(self, idx):
        hist_embedding = self.hist_embedding.iloc[idx]
        hist_label = self.hist_label.iloc[idx]
        query_embedding = self.query_embedding.iloc[idx].values
        category_idx = self.category.iloc[idx]
        input_vec = np.concatenate([hist_embedding, hist_label, query_embedding])

        return input_vec, category_idx

    def __len__(self):
        return len(self.category)

class HistoricalQueryDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size=512,
            num_workers=16
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.test = None
        self.valid = None

        self.label_encoding = {}

    def prepare_data(self):
        with open('datasets/aol_data_train_input_df.pkl', 'rb') as f:
            self.train = HistoricalModelDataset(pickle.load(f), self.label_encoding)
        with open('datasets/aol_data_test_input_df.pkl', 'rb') as f:
            self.test = HistoricalModelDataset(pickle.load(f), self.label_encoding)
        with open('datasets/aol_data_valid_input_df.pkl', 'rb') as f:
            valid_df = pickle.load(f)
            self.valid = HistoricalModelDataset(valid_df, self.label_encoding)

        self.label_encoding = {
            cat: i for i, cat in enumerate(
                valid_df['category'].unique()
            )
        }

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )