import logging
import os
os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from gpt2_model import GPT2SearchQueryDataModule, GPT2Classifier
from pathlib import Path

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('gpt2-logs/')
    N_DEVICES = 8
    encoding = {
        'Arts': 0,
        'Business': 11,
        'Computers': 10,
        'Games': 12,
        'Health': 9,
        'Home': 6,
        'News': 14,
        'Recreation': 1,
        'Reference': 13,
        'Regional': 4,
        'Science': 8,
        'Shopping': 3,
        'Society': 2,
        'Sports': 5,
        'World': 7
    }
    queries = GPT2SearchQueryDataModule(
        'open_source.feather',
        batch_size=128,
        num_workers=0,
        tokeniser_string='gpt2',
        debug=False,
        encoding=encoding,
    )
    queries.prepare_data()
    weights = queries.calculate_weights()
    model = GPT2Classifier(
        'gpt2',
        lr=1e-4,
        weights=weights,
        num_labels=15
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='valid/loss',
        dirpath='gpt2-checkpoints/',
        filename='model-{epoch:02d}-{valid/loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    Path('gpt2_model_save').mkdir(exists_ok=True)
    trainer = pl.Trainer(
        tpu_cores=N_DEVICES,
        max_epochs=5,
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        val_check_interval=2500,
        default_root_dir='gpt2_model_save'
    )
    trainer.fit(model, queries)