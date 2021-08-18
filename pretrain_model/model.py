import pytorch_lightning as pl
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from torch import nn, optim
import torchmetrics



class RobertaForPretraining(pl.LightningModule):
    def __init__(
            self,
            lr: float = 1e-4,
            adam_epsilon: float = 1e-6,
            adam_beta_1: float = 0.9,
            adam_beta_2: float = 0.98,
            weight_decay: float = 0.1,
            vocab_size: float = 49739,
            max_position_embeddings: float = 514,
            num_attention_heads: float = 12,
            num_hidden_layers: float = 6,
            type_vocab_size: float = 1,
            ):
        super().__init__()
        self.model_config = RobertaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
        )
        self.model = RobertaForMaskedLM(config=self.model_config)
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.adam_params = {
            'epsilon': adam_epsilon,
            'beta_1': adam_beta_1,
            'beta_2': adam_beta_2,
        }
        self.weight_decay = weight_decay

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss = self(input_ids, attention_mask, labels)['loss']
        self.log(
            'train/loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss = self(input_ids, attention_mask, labels)['loss']
        self.log(
            'valid/loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss = self(input_ids, attention_mask, labels)['loss']
        self.log(
            'test/loss',
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(
                self.adam_params['beta_1'],
                self.adam_params['beta_2']
            ),
            eps=self.adam_params['epsilon'],
            weight_decay=self.weight_decay
        )