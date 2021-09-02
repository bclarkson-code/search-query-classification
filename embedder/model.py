from transformers import GPT2ForSequenceClassification
from classifier import GPT2Classifier
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics


class Embedder(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str = "~/search-query-classification/pretrain_model/pretrain-checkpoints/model-epoch=01-val"
        "/loss=0.00.ckpt",
        lr: float = 1e-4,
        weights: list = None,
        num_labels=7,
        use_pretrained=True,
    ):
        super().__init__()
        transformer = GPT2Classifier.load_from_checkpoint(checkpoint_path)
        self.embedder = transformer.transformer.transformer

        if weights:
            weights = torch.tensor(weights)
            self.loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        return self.embedder(
            input_ids=input_ids, attention_mask=attention_mask
        )[0][:, -1, :]

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        return self(input_ids, attention_mask)
