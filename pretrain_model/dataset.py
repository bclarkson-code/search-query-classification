import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from transformers import RobertaTokenizerFast, LineByLineTextDataset, DataCollatorForLanguageModeling
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import ByteLevelBPETokenizer
from glob import glob
from pathlib import Path
from datasets import load_dataset, load_from_disk

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
            persistent_workers=True,

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
        if os.path.exists(f'datasets/train'):
            self.train = load_from_disk(f'datasets/train').set_format(type='torch', columns=[
                'input_ids', 'attention_mask'])
        else:
            self.train = None
        if os.path.exists(f'datasets/valid'):
            self.valid = load_from_disk(f'datasets/valid').set_format(type='torch', columns=[
                'input_ids', 'attention_mask'])
        else:
            self.valid = None
        if os.path.exists(f'datasets/test'):
            self.test = load_from_disk(f'datasets/test').set_format(type='torch', columns=[
                'input_ids', 'attention_mask'])
        else:
            self.test = None
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

    def prepare_dataset(self, dataset_path):
        ds_file = os.path.join(self.data_path, f'{dataset_path}.txt')
        dataset = load_dataset(
            "text",
            data_files=ds_file,
            split=['train'])[0]
        dataset = dataset.map(
            lambda ex: self.tokeniser(
                ex["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length),
            batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataset.save_to_disk(f'datasets/{dataset_path}')
        return dataset


    def prepare_data(self):
        # Build tokeniser
        if not os.path.exists(self.tokeniser_path):
            self.train_tokeniser()
        self.tokeniser = RobertaTokenizerFast.from_pretrained(
            self.tokeniser_path,
            max_len=self.max_length,
            truncation=True,
            padding='max_length'
        )
        if not os.path.exists(f'datasets/train'):
            self.train = self.prepare_dataset('train')
        if not os.path.exists(f'datasets/valid'):
            self.valid = self.prepare_dataset('valid')
        if not os.path.exists(f'datasets/test'):
            self.test = self.prepare_dataset('test')

        print(f'Train: {self.train}')
        print(f'Valid: {self.valid}')
        print(f'Test: {self.test}')

    def setup(self, stage=None):
        self.tokeniser = RobertaTokenizerFast.from_pretrained(
            self.tokeniser_path,
            max_len=self.max_length,
            truncation=True,
            padding='max_length'
        )
        print(help(DataCollatorForLanguageModeling))
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokeniser,
            mlm=True,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=24
        )

    def train_dataloader(self):
        print(f'self.train: {self.train}')
        dl = DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
        print(f'Train DataLoader: {dl}')
        print(f'Train DataSet: {dl.dataset}')
        return dl

    def val_dataloader(self):
        print(f'self.valid: {self.valid}')
        dl = DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
        print(f'Valid DataLoader: {dl}')
        print(f'Valid DataSet: {dl.dataset}')
        return dl

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
