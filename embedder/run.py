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
    queries = EmbedderData(
        "d:/Downloads",
        batch_size=4096,
        num_workers=os.cpu_count(),
    )
    embedder = Embedder(
        "d:/Downloads/gpt2_lightning_logs/lightning_logs/version_6/checkpoints/epoch=2-step=13999.ckpt",
    )
    trainer = pl.Trainer(
        gpus=2,
        precision=16,
    )
    Path("preds").mkdir(exist_ok=True)
    preds = trainer.predict(embedder, queries.debug_dataloader())
    print(f"Debug predictions: {preds}")
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
