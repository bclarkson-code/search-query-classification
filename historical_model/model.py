import torch
from torch import nn
import pytorch_lightning as pl

class HistoricalClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear((767 * 2 + 16), 512),
            nn.ReLU(),
            nn.Linear(512, 16)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.classifier(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer