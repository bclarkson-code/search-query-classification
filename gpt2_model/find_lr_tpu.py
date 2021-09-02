import logging
import os
from matplotlib import pyplot as plt

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
        # gpus=2,
        max_epochs=1,
        progress_bar_refresh_rate=1,
        val_check_interval=2000,
        # precision=16,
        # accelerator="ddp",
    )

    lr_finder = trainer.tuner.lr_find(model, queries)

    # Results can be found in
    lr_finder.results

    with open("lr_finder_results", "wb") as f:
        pickle.dump(lr_finder.results, f)
    # trainer.fit(model, queries)
