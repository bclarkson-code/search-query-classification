import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from transformers import RobertaTokenizerFast, LineByLineTextDataset, DataCollatorForLanguageModeling
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import ByteLevelBPETokenizer
from glob import glob
from pathlib import Path
from datasets import load_dataset

class SearchQueryPreTrainingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str = "text_datasets",
            tokeniser_path: str = "tokeniser",
            batch_size: int = 128,
            num_workers: int = None,
            max_length: int = 24,
            mlm_probability: float = 0.15,
            debug=False,
            persistent_workers=False,

    ):

        super().__init__()
        self.data_path = data_path
        self.tokeniser_path = tokeniser_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.debug = debug
        self.persistent_workers = persistent_workers
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
        self.tokeniser = ByteLevelBPETokenizer(lowercase=True)
        self.tokeniser.pre_tokenizer = Whitespace()
        self.tokeniser.train(
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
        self.tokeniser.save_model(self.tokeniser_path)

    def prepare_data(self):
        # Build tokeniser
        if not os.path.exists(self.tokeniser_path):
            self.train_tokeniser()
        self.tokeniser = RobertaTokenizerFast.from_pretrained(
            self.tokeniser_path,
            max_len=self.max_length,
        )

        # Build datasets
        for ds_name in ['train', 'test', 'valid']:
            ds = self.__dict__[ds_name]
            if ds is None:
                print(f'Reading {ds_name.capitalize()} Data...', end='')
                ds_file = os.path.join(self.data_path, 'valid.txt')#f'{ds_name}.txt')
                self.__dict__[ds_name] = load_dataset(
                    "text",
                    data_files=ds_file,
                    split=['train'])[0]
                print(self.__dict__[ds_name])
                self.__dict__[ds_name] = self.__dict__[ds_name].map(
                    lambda ex: self.tokeniser(
                        ex["text"],
                        add_special_tokens=True,
                        truncation=True,
                        max_length=self.max_length),
                    batched=True)
                self.__dict__[ds_name].set_format(type='torch', columns=['input_ids',
                                                                         'attention_mask'])
                print('Done')

    def setup(self, stage=None):
        self.tokeniser = RobertaTokenizerFast.from_pretrained(
            self.tokeniser_path,
            max_len=self.max_length,
        )
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
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        print(len(self.test))
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

if __name__ == '__main__':
    ds = SearchQueryPreTrainingDataModule()
    ds.prepare_data()
