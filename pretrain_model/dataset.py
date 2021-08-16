import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from transformers import RobertaTokenizerFast, LineByLineTextDataset, DataCollatorForLanguageModeling
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import ByteLevelBPETokenizer
from glob import glob
from pathlib import Path

class SearchQueryPreTrainingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str = "text_datasets",
            tokeniser_path: str = "tokeniser",
            batch_size: int = 128,
            num_workers: int = None,
            max_length: int = 24,
            mlm_probability: float = 0.15,
            debug=False

    ):

        super().__init__()
        self.data_path = data_path
        self.tokeniser_path = tokeniser_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.debug = debug
        if num_workers is None:
            self.num_workers = os.cpu_count()
        else:
            self.num_workers = num_workers
        self.tokeniser = None
        self.train = None
        self.test = None
        self.valid = None
        self.data_collator = None

    def train_tokeniser(self):
        self.tokenizer = ByteLevelBPETokenizer(lowercase=True)
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train(
            files=glob(f'{self.data_path}/all.txt'),
            vocab_size=50000,
            min_frequency=2,
            show_progress=True,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ])
        Path(self.tokeniser_path).mkdir(exist_ok=True)
        self.tokenizer.save_model(self.tokeniser_path)

    def prepare_data(self):
        # Build tokeniser
        vocab_path = os.path.abspath(os.path.join(self.tokeniser_path, 'vocab.json'))
        merges_path = os.path.abspath(os.path.join(self.tokeniser_path, 'merges.txt'))
        if not os.path.exists(vocab_path) and not os.path.exists(vocab_path):
            self.train_tokeniser()
        else:
            self.tokeniser = ByteLevelBPETokenizer(vocab_path, merges_path)

        # Build datasets
        for ds_name in ['train', 'test', 'valid']:
            ds = self.__dict__[ds_name]
            if ds is None:
                print(f'Reading {ds_name.capitalize()} Data...', end='')
                ds_file = os.path.join(self.data_path, f'{ds_name}.txt')
                self.__dict__[ds_name] = LineByLineTextDataset(
                    tokenizer=self.tokeniser,
                    file_path=ds_file,
                    block_size=128,
                )
                print('Done')

    def setup(self, stage=None):
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokeniser,
            mlm=True,
            mlm_probability=self.mlm_probability
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.num_workers
        )

if __name__ == '__main__':
    ds = SearchQueryPreTrainingDataModule()
    ds.prepare_data()
