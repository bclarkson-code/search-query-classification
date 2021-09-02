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

if __name__ == "__main__":
    queries = EmbedderData(
        "/home/benedictclarkson1/search-query-classification/gpt2_model",
        batch_size=4096,
        num_workers=os.cpu_count(),
    )
    embedder = Embedder(
        "/home/benedictclarkson1/search-query-classification/gpt2_model/lightning_logs/version_6/checkpoints/epoch=2-step=13999.ckpt",
    )
    trainer = pl.Trainer(
        tpu_cores=8,
        precision=16,
    )
    with torch.no_grad():
        embedder.eval()
        for loader, ds_name in zip(
            [
                queries.val_dataloader(),
                queries.train_dataloader(),
                queries.test_dataloader(),
            ],
            ["valid", "train", "test"],
        ):
            preds = trainer.predict(embedder, loader)
            with open(f"preds/{ds_name}_preds.npy", "wb") as f:
                np.save(f, preds)
