import logging
import os

# Variables to speed up TPU
os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from model import GPT2Classifier
from dataset import SearchQueryDataModule

if __name__ == "__main__":
    N_DEVICES = 8
    queries = SearchQueryDataModule(batch_size=128)
    model = GPT2Classifier("gpt2", lr=1e-1, num_labels=15)

    trainer = pl.Trainer(
        tpu_cores=N_DEVICES,
        max_epochs=3,
        progress_bar_refresh_rate=1,
        val_check_interval=2000,
        precision=16,
        limit_train_batches=5000,
        limit_val_batches=100,
    )

    trainer.fit(model, queries)
