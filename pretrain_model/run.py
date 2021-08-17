import logging
import os
os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SearchQueryPreTrainingDataModule
import dataset, model
from model import RobertaForPretraining

if __name__ == '__main__':
    print(f'model_path: {model.__file__}')
    print(f'dataset_path: {dataset.__file__}')
    tb_logger = pl_loggers.TensorBoardLogger('pretrain-logs/')
    N_DEVICES = 8
    lr = 5e-5
    batch_size = 128
    print(f'lr: {lr}')
    print(f'batch_size: {batch_size}')
    queries = SearchQueryPreTrainingDataModule(
        batch_size=128,
        debug=True,
        num_workers=os.cpu_count(),
        persistent_workers=True
    )
    model = RobertaForPretraining(
        lr=5e-5
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='valid/loss',
        dirpath='pretrain-checkpoints/',
        filename='model-{epoch:02d}-{val/loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    trainer = pl.Trainer(
        gpus=4,
        max_epochs=3,
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        auto_lr_find=True,
        callbacks=checkpoint_callback,
        val_check_interval=2500,
        accelerator='ddp',
    )
    trainer.fit(model, queries)