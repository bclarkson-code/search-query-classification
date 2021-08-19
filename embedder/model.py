from pretrain_model import RobertaForPretraining
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics

class Classifier(pl.LightningModule):
    def __init__(
            self,
            checkpoint_path: str =
            '~/search-query-classification/pretrain_model/pretrain-checkpoints/model-epoch=01-val'
            '/loss=0.00.ckpt',
            lr: float = 1e-4,
            weights: list = None,
            num_labels=7):
        super().__init__()
        transformer = RobertaForPretraining.load_from_checkpoint(checkpoint_path)
        self.embedder = transformer.model.roberta
        self.classifier = nn.Linear(768, num_labels)

        if weights:
            weights = torch.tensor(weights)
            self.loss = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask):
        embedding = self.embedder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        input_ids, attention_mask = inputs
        input_ids = torch.stack(input_ids).T
        attention_mask = torch.stack(attention_mask).T
        preds = self(input_ids, attention_mask)
        loss = self.loss(preds, targets)
        self.log(
            'train-loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        self.train_acc(preds, targets)

        return loss

    def training_epoch_end(self, outputs):
        self.log(
            'train-accuracy',
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        input_ids, attention_mask = inputs
        input_ids = torch.stack(input_ids).T
        attention_mask = torch.stack(attention_mask).T
        preds = self(input_ids, attention_mask)
        loss = self.loss(preds, targets)
        self.log(
            'valid-loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        self.valid_acc(preds, targets)

    def validation_epoch_end(self, outputs):
        self.log(
            'valid-accuracy',
            self.valid_acc.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

    def configure_optimizers(self):
        optimiser = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimiser,
            max_lr=1e-4,
            total_steps=70455,
        )
        return {
            'optimizer': optimiser,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
