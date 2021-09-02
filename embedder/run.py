import logging
import os

os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import torch
import torch_xla.core.xla_model as xm
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
    device = xm.xla_device()
    with torch.no_grad():
        embedder.to(device)
        embedder.eval()
        for loader, ds_name in zip(
            [
                queries.val_dataloader(),
                queries.train_dataloader(),
                queries.test_dataloader(),
            ],
            ["valid", "train", "test"],
        ):
            for i, batch in tqdm(
                loader, desc=f"Embedding {ds_name}", total=len(loader)
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                preds = embedder.predict_step(batch)
                preds = preds.cpu().numpy()
                with open(f"preds/valid_preds_{i}.npy", "wb") as f:
                    np.save(f, preds)
