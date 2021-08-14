import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import subprocess
import shlex
from pathlib import Path
from transformers import RobertaTokenizerFast, LineByLineTextDataset, DataCollatorForLanguageModeling


class SearchQueryPreTrainingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str = "open_source_text_datasets",
            tokeniser_path: str = 'open_source_tokeniser',
            batch_size: int = 128,
            num_workers: int = None,
            max_length: int = 24,
            gs_bucket_path: str = 'gs://search-query-classification-us-central1-c/',
            mlm_probability: float = 0.15,
            debug=False
    ):

        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.gs_bucket_path = gs_bucket_path
        self.mlm_probability = mlm_probability
        self.tokeniser_path = tokeniser_path
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

    def _download_file(self, file_path, print_output=False):
        """
        Download a file from gcs
        :param file_path: the path of the file in the gcs bucket
        """
        zip_path = file_path + '.zip'
        gsutil_path = os.path.join(self.gs_bucket_path, zip_path)
        download_cmd = shlex.split(
            f'gsutil cp {gsutil_path} .'
        )
        unzip_cmd = shlex.split(
            f'unzip -o {zip_path}'
        )
        clean_up_cmd = shlex.split(
            f'rm -f {zip_path}'
        )
        commands = [download_cmd, unzip_cmd, clean_up_cmd]
        for cmd in commands:
            if print_output:
                print(subprocess.check_output(
                    cmd,
                    stderr=subprocess.STDOUT).decode('utf-8')
                )
            else:
                subprocess.call(cmd)

    def _download_tokeniser(self):
        print('Downloading Tokeniser...', end='')
        self._download_file(self.tokeniser_path)
        assert os.path.exists(self.tokeniser_path)
        print('Done')

    def _download_data(self):
        print('Downloading Data...', end='')
        self._download_file(self.data_path, print_output=self.debug)
        assert os.path.exists(self.data_path)
        print('Done')
        # if debug, only take the first 500 lines of each text file
        if self.debug:
            print()
            for ds_name in ['train', 'test', 'valid']:
                ds_file = os.path.join(self.data_path, f'{ds_name}.txt')
                with open(ds_file, 'r') as f:
                    data = [line for line in f]
                with open(ds_file, 'w') as f:
                    f.write('\n'.join(data[:5000]))


    def prepare_data(self):
        # Build tokeniser
        if not os.path.exists(self.tokeniser_path):
            self._download_tokeniser()
        self.tokeniser = RobertaTokenizerFast.from_pretrained(
            self.tokeniser_path,
            max_len=self.max_length
        )
        # Download raw data
        self._download_data()

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
