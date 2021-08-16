import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics


class HistoricalClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear((767 * 2 + 16), 512),
            torch.nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 16)
        )
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(
            'train_loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        self.train_acc(pred, y)
        return loss

    def training_epoch_end(self, outputs):
        self.log(
            'train_accuracy',
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(
            'valid_loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        self.valid_acc(pred, y)
        return loss

    def validation_epoch_end(self, outputs):
        self.log(
            'valid_accuracy',
            self.valid_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=5e-3, # base and max lr found from lr finder
            max_lr=1e-1,
            step_size_up=117039 # batches per epoch
        )
        return [optimizer], [scheduler]
