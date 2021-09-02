import logging
import os

os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_TENSOR_ALLOCATOR_MAXSIZE"] = "100000000"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from model import Classifier
from dataset import EmbedderData
import pickle

if __name__ == "__main__":
    queries = EmbedderData(
        batch_size=2048,
        num_workers=os.cpu_count(),
    )
    embedder = Embedder()
    trainer = pl.Trainer(
        tpu_cores=8,
        precision=16,
    )
    val_preds = trainer.predict(embedder, data.val_dataloader())
    val_preds = val_pred.detach().numpy()
    np.save(val_preds)
