from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from transformers import GPT2ForSequenceClassification


class GPT2Classifier(pl.LightningModule):
    def __init__(
        self,
        transformer_string: str = "gpt2",
        lr: float = 1e-4,
        weights: list = None,
        num_labels=15,
    ):
        super().__init__()
        self.transformer = GPT2ForSequenceClassification.from_pretrained(
            transformer_string,
            num_labels=num_labels,
        )
        self.transformer.config.pad_token_id = 50256
        if weights:
            weights = torch.tensor(weights)
            self.loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        return self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = [
            batch[col] for col in ["input_ids", "attention_mask", "category"]
        ]

        preds = self(input_ids, attention_mask)
        loss = self.loss(preds, targets)
        self.log("train/loss", loss, on_step=True, sync_dist=True)
        self.train_acc(preds, targets)

        return loss

    def training_epoch_end(self, outputs):
        self.log(
            "train/accuracy",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = [
            batch[col] for col in ["input_ids", "attention_mask", "category"]
        ]
        preds = self(input_ids, attention_mask)
        loss = self.loss(preds, targets)
        self.log(
            "valid/loss", loss, on_step=True, on_epoch=False, sync_dist=True
        )
        self.valid_acc(preds, targets)

    def validation_epoch_end(self, outputs):
        self.log(
            "valid/accuracy",
            self.valid_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim
