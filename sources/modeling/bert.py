import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from typing import List
from .base.tokenizer import BaseTokenizer


class BERT(BaseTokenizer):

    # hyperparameters
    BATCH_SIZE = 32
    GRADIENT_N_ACCUMULATION_STEPS = 2
    N_EPOCHS = 5
    WARMUP_RATIO = 0.1
    LEARNING_RATE = 3e-5
    ADAM_EPSILON = 1e-6
    GRADIENT_MAX_NORM = 1

    def __init__(self):
        self.transformer_name = "neuralmind/bert-large-portuguese-cased"
        BaseTokenizer.__init__(self, self.transformer_name)

    def fit(self, samples: List[dict], y: np.ndarray) -> None:
        samples_tokenized = self._tokenizer_transform(samples)
        data_loader = self._create_data_loader(samples_tokenized, y)
        device = self._get_device()
        self._create_model()
        self._set_up_optimizers(data_loader)
        self._fit(data_loader, device)

    # def predict(self, samples: List[dict]) -> np.ndarray:
    #     tokens = self._tokenize()
    #     self._predict(self, x, y)

    def _tokenizer_transform(self, samples):
        return BaseTokenizer.transform(self, samples)

    @staticmethod
    def _get_device():
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available.")
        return torch.device("cuda:0")

    def _create_data_loader(self, samples, y):
        data = []
        for sample, label in zip(samples, y):
            data.append(sample)
            data[-1]["label"] = label
        return DataLoader(
            data,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_fn,
            drop_last=False
        )

    def _create_model(self):
        self.model = BaseBERT(self.transformer_name)
        self.model.to(0)
        self.model._encoder.resize_token_embeddings(len(self.tokenizer))

    def _set_up_optimizers(self, data_loader):
        num_training_steps = int(
            len(data_loader) * self.N_EPOCHS // self.GRADIENT_N_ACCUMULATION_STEPS)
        num_warmup_steps = int(num_training_steps * self.WARMUP_RATIO)
        self._optimizer = AdamW(
            self.model.parameters(), lr=self.LEARNING_RATE, eps=self.ADAM_EPSILON)
        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self._scaler = GradScaler()

    def _fit(self, data_loader, device):
        num_steps = 0
        for _ in range(self.N_EPOCHS):
            self.model.zero_grad()
            for step, batch in enumerate(tqdm(data_loader)):

                # train model
                self.model.train()
                inputs = {
                    "tokens": batch[0].to(device),
                    "attention_mask": batch[1].to(device),
                    "labels": batch[2].to(device),
                    "indexes_1": batch[3].to(device),
                    "indexes_2": batch[4].to(device),
                }
                outputs = self.model(**inputs)

                # compute loss
                loss = outputs[0] / self.GRADIENT_N_ACCUMULATION_STEPS
                self._scaler.scale(loss).backward()

                # optimize
                if step % self.GRADIENT_N_ACCUMULATION_STEPS == 0:
                    num_steps += 1

                    # clip gradient
                    if self.GRADIENT_MAX_NORM > 0:
                        self._scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.GRADIENT_MAX_NORM)

                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._scheduler.step()
                    self.model.zero_grad()

    @staticmethod
    def _collate_fn(batch):
        max_len = max([len(x["tokens"]) for x in batch])

        tokens = [x["tokens"] + [0] * (max_len - len(x["tokens"])) for x in batch]
        input_mask = [
            [1.0] * len(x["tokens"]) + [0.0] * (max_len - len(x["tokens"]))
            for x in batch
        ]
        labels = [x["label"] for x in batch]
        indexes_1 = [x["index_1"] for x in batch]
        indexes_2 = [x["index_2"] for x in batch]

        tokens = torch.tensor(tokens, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        indexes_1 = torch.tensor(indexes_1, dtype=torch.long)
        indexes_2 = torch.tensor(indexes_2, dtype=torch.long)

        return tokens, input_mask, labels, indexes_1, indexes_2


class BaseBERT(nn.Module):

    # hyperparameters
    N_LABELS = 2
    DROPOUT_RATE = 0.1

    def __init__(self, transformer_name):
        super().__init__()
        config = AutoConfig.from_pretrained(transformer_name, num_labels=self.N_LABELS)
        config.gradient_checkpointing = True
        self._encoder = AutoModel.from_pretrained(transformer_name, config=config)
        self._loss_function = nn.CrossEntropyLoss()
        self._classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.DROPOUT_RATE),
            nn.Linear(config.hidden_size, self.N_LABELS)
        )

    @autocast()
    def forward(
            self,
            tokens=None,
            attention_mask=None,
            labels=None,
            indexes_1=None,
            indexes_2=None
    ):
        outputs = self._encoder(
            tokens,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]

        idx = torch.arange(tokens.size(0)).to(tokens.device)
        indexes_1_emb = pooled_output[idx, indexes_1]
        indexes_2_emb = pooled_output[idx, indexes_2]
        h = torch.cat((indexes_1_emb, indexes_2_emb), dim=-1)
        logits = self._classifier(h)
        outputs = (logits,)

        if labels is not None:
            loss = self._loss_function(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs
