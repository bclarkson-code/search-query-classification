import logging
import os
os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from model import Classifier
from dataset import EmbedderData
import pickle

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('embedder-logs/')
    N_DEVICES = 8
    queries = EmbedderData(
        batch_size=256,
        num_workers=os.cpu_count(),
    )
    queries.prepare_data()
    queries.setup()
    weights = queries.calculate_weights()
    classifier = Classifier(
        lr=1e-4,
        weights=weights,
        num_labels=7
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='valid-loss',
        dirpath='embedder-checkpoints/',
        filename='model-{epoch:02d}-{valid-loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    trainer = pl.Trainer(
        tpu_cores=8,
        max_epochs=3,
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        auto_lr_find=True,
        callbacks=checkpoint_callback,
        val_check_interval=2500,
        limit_val_batches=0.1,
        precision=16,
    )
    trainer.fit(classifier, queries)