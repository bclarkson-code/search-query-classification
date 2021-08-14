from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class GPT2TextDataset(Dataset):
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

class GPT2SearchQueryDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str = "raw_data.pkl",
            batch_size: int = 2,
            num_workers: int = 1,
            encoding: dict = {},
            tokeniser_string: str = 'gpt2',
            debug: bool = False,
            data_frac=0.1,

    ):

        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokeniser = AutoTokenizer.from_pretrained(tokeniser_string)
        self.data = None
        self.debug = debug
        self.data_frac = data_frac
        self.classes = None
        if not encoding:
            unique_classes = ['TRACK', 'EPISODE', 'PLAYLIST_V2', 'ALBUM', 'ARTIST', 'SHOW', 'USER']
            self.encoding = {cls: i for i, cls in enumerate(unique_classes)}
        else:
            self.encoding = encoding

    def prepare_data(self):
        if self.data is None:
            print('Reading Data...', end='')
            file_extension = self.data_path.split('.')[-1]
            if file_extension == 'pkl':
                self.data = pd.read_pickle(self.data_path)
            elif file_extension == 'feather':
                self.data = pd.read_feather(self.data_path)
            else:
                raise IOError(f'File format: {file_extension} is not supported')
            print('Done')
            print('Pre-Processing...', end='')
            if self.debug:
                self.data = self.data.sample(frac=self.data_frac)
        self.classes = self.data['class'].unique()
        print('Done')

    def setup(self, stage=None):
        self.data = self.data.reset_index()
        train = self.data[self.data.index % 10 > 1]
        test = self.data[self.data.index % 10 == 0]
        valid = self.data[self.data.index % 10 == 1]
        print(f'All data: {len(self.data)} lines')
        print(f'Train data: {len(train)} lines')
        print(f'Test data: {len(test)} lines')
        print(f'Valid data: {len(valid)} lines')

        self.train = GPT2TextDataset(
            train,
            self.tokeniser,
            self.encoding
        )

        self.test = GPT2TextDataset(
            test,
            self.tokeniser,
            self.encoding
        )

        self.valid = GPT2TextDataset(
            valid,
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
        class_freq = dict(self.data['class'].value_counts())
        class_freq = {self.encoding[key]: val for key, val in class_freq.items()}
        counts = [(key, val) for key, val in class_freq.items()]
        counts = sorted(counts)
        counts = [c[1] for c in counts]
        return [float(sum(counts)/val) for val in counts]

class GPT2Classifier(pl.LightningModule):
    def __init__(
            self,
            transformer_string: str = 'gpt2',
            lr: float = 1e-4,
            weights: list = None,
            num_labels=15):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            transformer_string,
            num_labels=num_labels
        )
        self.transformer.config.pad_token_id = 50256
        if weights:
            weights = torch.tensor(weights)
            self.loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        input_ids, attention_mask = inputs

        preds = self(input_ids, attention_mask)
        loss = self.loss(preds, targets)
        self.log(
            'train/loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        self.train_acc(preds, targets)

        return loss

    def training_epoch_end(self, outputs):
        self.log(
            'train/accuracy',
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        input_ids, attention_mask = inputs
        preds = self(input_ids, attention_mask)
        loss = self.loss(preds, targets)
        self.log(
            'valid/loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        self.valid_acc(preds, targets)

    def validation_epoch_end(self, outputs):
        self.log(
            'valid/accuracy',
            self.valid_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
