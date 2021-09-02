import logging
import os

os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import torch
import pytorch_lightning as pl
from dataset import EmbedderData
from model import Embedder
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

if __name__ == "__main__":
    with torch.no_grad():
        queries = EmbedderData(
            "/home/benedict/search-query-classification/gpt2_model/datasets",
            batch_size=4096,
            num_workers=os.cpu_count(),
        )
        embedder = Embedder(
            "/home/benedict/search-query-classification/gpt2_model/lightning_logs/version_6/checkpoints/epoch=2-step=13999.ckpt",
        )
        embedder.eval()

        trainer = pl.Trainer(tpu_cores=8, precision=16)
        os.system("rm -rf preds")
        Path("preds").mkdir(exist_ok=True)
        debug = queries.debug_dataloader()
        preds = trainer.predict(embedder, debug)
        print(f"Debug succeeded: {preds is not None}")

        for loader in [
            queries.val_dataloader(),
            queries.train_dataloader(),
            queries.test_dataloader(),
        ]:
            trainer.predict(embedder, loader)
