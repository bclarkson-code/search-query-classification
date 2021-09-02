import logging
import os

os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from dataset import EmbedderData
from model import Embedder
import pickle

if __name__ == "__main__":
    queries = EmbedderData(
        "/home/benedictclarkson1/search-query-classification/gpt2_model",
        batch_size=2048,
        num_workers=os.cpu_count(),
    )
    embedder = Embedder(
        "/home/benedictclarkson1/search-query-classification/gpt2_model/lightning_logs/version_6/checkpoints/epoch=2-step=13999.ckpt",
    )
    trainer = pl.Trainer(
        tpu_cores=8,
        precision=16,
    )
    val_preds = trainer.predict(embedder, queries.val_dataloader())
    val_preds = val_preds.detach().numpy()
    np.save(val_preds, "valid_preds")

    train_preds = trainer.predict(embedder, queries.train_dataloader())
    train_preds = train_preds.detach().numpy()
    np.save(train_preds, "train_preds")

    test_preds = trainer.predict(embedder, queries.test_dataloader())
    test_preds = test_preds.detach().numpy()
    np.save(test_preds, "test_preds")
