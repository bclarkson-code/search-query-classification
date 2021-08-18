from transformers import RobertaTokenizerFast
import os
import shlex
import subprocess
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class Downloader(object):
    def __init__(self, data_dir='datasets'):
        self.data_dir = data_dir

    def download_datasets(self):
        Path('raw_dataset').mkdir(exist_ok=True)

        # Download the raw dataset from google cloud storage
        cmd = shlex.split(
            'gsutil -m cp gs://search-query-classification-us-central1-c/raw_dataset/* raw_dataset'
        )
        print(subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT).decode('utf-8')
              )

        df = pd.read_parquet('raw_dataset')

        Path('dataframes').mkdir(exist_ok=True)
        train_frac = 0.8
        valid_frac = 0.1
        n_queries = len(df)
        datasets = np.split(
            df.sample(frac=1, random_state=42),
            [
                int(train_frac * n_queries),
                int((train_frac + valid_frac) * n_queries)
            ]
        )
        datasets.append(df)
        datasets.append(df.sample(n=1000))
        for ds, ds_name in zip(datasets, ['train', 'test', 'valid', 'all', 'debug']):
            ds.to_pickle(f'dataframes/{ds_name}.pkl')

    def download_tokeniser(self):
        # Download the tokeniser zip file from google cloud storage
        cmd = shlex.split(
            'gsutil -m cp gs://search-query-classification-europe-west4-a/tokeniser.zip .'
        )
        print(subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT).decode('utf-8')
              )

        # unzip the tokeniser folder
        cmd = shlex.split(
            'unzip tokeniser.zip'
        )
        print(subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT).decode('utf-8')
              )


class TextDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokeniser,
                 label_encoding: dict):
        print('Tokenising...', end='')
        self.tokeniser = tokeniser
        self.tokeniser.pad_token = self.tokeniser.eos_token
        text = df['query_text'].to_list()
        self.tokens = self._tokenise(text)
        print('Done')
        self.labels = df['class'].values
        self.label_encoding = label_encoding

    def __getitem__(self, idx):
        keys = ['input_ids', 'attention_mask']
        tokens = [self.tokens[key][idx] for key in keys]
        cls = self.label_encoding[self.labels[idx]]
        return tokens, cls

    def __len__(self):
        return len(self.labels)

    def _tokenise(self, string):
        tokens = self.tokeniser(
            string,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=24
        )
        return tokens.__dict__['data']

class EmbedderData(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str = "dataframes",
            batch_size: int = 2,
            num_workers: int = 1,
            encoding: dict = {},
            max_length: int = 24,
            pretrain_path='~/search-query-classification/pretrain_model',
            tokeniser_string='tokeniser',
            classes=['TRACK', 'EPISODE', 'PLAYLIST_V2', 'ALBUM', 'ARTIST', 'SHOW', 'USER']

    ):

        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.pretrain_path = pretrain_path
        self.downloader = Downloader()

        # Read tokeniser from pretrain model
        if not os.path.exists(tokeniser_string):
            self.downloader.download_tokeniser()
        self.tokeniser = RobertaTokenizerFast.from_pretrained(
            tokeniser_string,
            max_len=self.max_length,
            truncation=True,
            padding='max_length'
        )

        self.classes = classes
        if not encoding:
            self.encoding = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.encoding = encoding


    def prepare_data(self):
        if not os.path.exists('dataframes'):
            print('Downloading Data...', end='')
            self.downloader.download_datasets()
            print('Done')

    def setup(self, stage=None):
        if 'train' not in self.__dict__:
            self.train = pd.read_pickle('dataframes/train.pkl')
            self.valid = pd.read_pickle('dataframes/valid.pkl')
            self.test = pd.read_pickle('dataframes/test.pkl')
        print(f'Train data: {len(self.train)} lines')
        print(f'Test data: {len(self.test)} lines')
        print(f'Valid data: {len(self.valid)} lines')

        self.train = TextDataset(
            self.train,
            self.tokeniser,
            self.encoding
        )

        self.test = TextDataset(
            self.test,
            self.tokeniser,
            self.encoding
        )

        self.valid = TextDataset(
            self.valid,
            self.tokeniser,
            self.encoding
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def calculate_weights(self):
        class_freq = dict(self.train['class'].value_counts())
        class_freq = {self.encoding[key]: val for key, val in class_freq.items()}
        counts = [(key, val) for key, val in class_freq.items()]
        counts = sorted(counts)
        counts = [c[1] for c in counts]
        return [float(sum(counts)/val) for val in counts]

