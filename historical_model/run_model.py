from model import HistoricalClassifier
import pytorch_lightning as pl
from pathlib import Path
from dataset import HistoricalQueryDataModule
from pytorch_lightning import loggers as pl_loggers
import os
import pickle

if __name__ == '__main__':
    tb_logger = pl_loggers.TensorBoardLogger('historical-model-logs/')
    inputs = HistoricalQueryDataModule(
        batch_size=128,
        num_workers=os.cpu_count(),
    )
    classifier = HistoricalClassifier(
        learning_rate=1e-4,
    )
    Path('model_save').mkdir(exist_ok=True)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        progress_bar_refresh_rate=1,
        logger=tb_logger,
        val_check_interval=2500,
        default_root_dir='model_save'
    )
    lr_finder = trainer.tuner.lr_find(classifier, inputs)
    with open('lr_finder_results.pkl', 'wb') as f:
        pickle.dump(lr_finder.results, f)

    trainer.fit(classifier, inputs)