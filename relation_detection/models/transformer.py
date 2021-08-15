import numpy as np
import random
import torch
import torch.nn as nn
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, logging
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from typing import List, Optional, Tuple
from .base.engineering import BaseEngineering
from .base.tokenizer import BaseTokenizer

# disable warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class Transformer(BaseEngineering, BaseTokenizer):

    # hyperparameters
    BATCH_SIZE = 32
    GRADIENT_N_ACCUMULATION_STEPS = 2
    N_EPOCHS = 5
    WARMUP_RATIO = 0.1
    LEARNING_RATE = 3e-5
    ADAM_EPSILON = 1e-6
    GRADIENT_MAX_NORM = 1

    def __init__(self) -> None:
        self.add_features_ = True
        self.transformer_name_ = "neuralmind/bert-large-portuguese-cased"
        BaseEngineering.__init__(self)
        BaseTokenizer.__init__(self, self.transformer_name_)
        if not torch.cuda.is_available():
            print("WARNING: GPU not found.")

    def fit(  # type: ignore[override]
            self,
            samples: List[dict],
            y: np.ndarray,
            groups: List[str]
    ) -> None:
        samples_tokenized = self._tokenizer_transform(samples)

        features_size = 0
        if self.add_features_:
            samples_tokenized = self._add_features(samples, samples_tokenized, fit=True)
            features_size = len(samples_tokenized[0]["features"])

        data_loader = self._create_data_loader(samples_tokenized, y, shuffle=True)
        device = self._get_device()
        self._create_model(device, features_size)
        self._set_up_optimizers(data_loader)
        self._fit(data_loader, device)

    def predict(
            self,
            samples: List[dict],
            for_lime: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        pad = True if for_lime else False
        samples_tokenized = self._tokenizer_transform(samples, pad)
        if self.add_features_:
            samples_tokenized = self._add_features(samples, samples_tokenized)
        predictions_proba = self._predict_proba(samples_tokenized)
        predictions = predictions_proba.argmax(axis=1)
        return predictions, predictions_proba

    def _predict_proba(self, samples_tokenized: List[dict]) -> np.ndarray:
        data_loader = self._create_data_loader(samples_tokenized)
        device = self._get_device()
        predictions_proba = []
        for batch in data_loader:

            self.model_.eval()
            inputs = {
                "tokens": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "indexes_1": batch[3].to(device),
                "indexes_2": batch[4].to(device),
                "features": batch[5].to(device)
            }
            with torch.no_grad():
                logit = self.model_(**inputs)[0]
            predictions_proba += nn.functional.softmax(logit, dim=-1).tolist()

        return np.array(predictions_proba)

    def _tokenizer_transform(self, samples: List[dict], pad: bool = False) -> List[dict]:
        return BaseTokenizer.transform(self, samples, pad)

    def _add_features(
            self,
            samples: List[dict],
            samples_tokenized: List[dict],
            fit: bool = False
    ) -> List[dict]:
        if fit:
            BaseEngineering.fit(self, samples)
        features = BaseEngineering.get_features(self, samples).to_numpy()
        new_samples = []
        for i in range(len(samples)):
            new_sample = {key: value for key, value in samples_tokenized[i].items()}
            new_sample["features"] = features[i, :]
            new_samples.append(new_sample)
        return new_samples

    @staticmethod
    def _get_device() -> torch.device:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        return device

    def _create_data_loader(
            self,
            samples: List[dict],
            y: Optional[np.ndarray] = None,
            shuffle: bool = False
    ) -> DataLoader:
        data = samples
        if y is not None:
            for sample, label in zip(data, y):
                sample["label"] = label
        return DataLoader(
            data,  # type: ignore
            batch_size=self.BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )

    def _create_model(self, device: torch.device, n_extra_features: int) -> None:
        self.model_ = BaseTransformer(self.transformer_name_, n_extra_features)
        self.model_.to(device=device)
        self.model_._encoder.resize_token_embeddings(len(self.tokenizer_))

    def _set_up_optimizers(self, data_loader: DataLoader) -> None:
        num_training_steps = int(
            len(data_loader) * self.N_EPOCHS // self.GRADIENT_N_ACCUMULATION_STEPS)
        num_warmup_steps = int(num_training_steps * self.WARMUP_RATIO)
        self._optimizer = AdamW(
            self.model_.parameters(), lr=self.LEARNING_RATE, eps=self.ADAM_EPSILON)
        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self._scaler = GradScaler()

    def _fit(self, data_loader: DataLoader, device: torch.device) -> None:
        num_steps = 0
        for _ in range(self.N_EPOCHS):
            self.model_.zero_grad()
            for step, batch in enumerate(data_loader):

                # train model
                self.model_.train()
                inputs = {
                    "tokens": batch[0].to(device),
                    "attention_mask": batch[1].to(device),
                    "labels": batch[2].to(device),
                    "indexes_1": batch[3].to(device),
                    "indexes_2": batch[4].to(device),
                    "features": batch[5].to(device)
                }
                outputs = self.model_(**inputs)

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
                            self.model_.parameters(), self.GRADIENT_MAX_NORM)

                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._scheduler.step()
                    self.model_.zero_grad()

    @staticmethod
    def _collate_fn(batch: list) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor]
    ]:
        max_len = max([len(x["tokens"]) for x in batch])

        tokens = [x["tokens"] + [0] * (max_len - len(x["tokens"])) for x in batch]
        input_mask = [
            [1.0] * len(x["tokens"]) + [0.0] * (max_len - len(x["tokens"]))
            for x in batch
        ]
        indexes_1 = [x["index_1"] for x in batch]
        indexes_2 = [x["index_2"] for x in batch]

        labels = None
        if "label" in batch[0]:
            labels = [x["label"] for x in batch]

        features = None
        if "features" in batch[0]:
            features = [x["features"] for x in batch]

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.float),
            torch.tensor(labels, dtype=torch.long) if labels is not None else None,
            torch.tensor(indexes_1, dtype=torch.long),
            torch.tensor(indexes_2, dtype=torch.long),
            torch.tensor(features, dtype=torch.float) if features is not None else None
        )


class BaseTransformer(nn.Module):

    # hyperparameters
    N_LABELS = 2
    DROPOUT_RATE = 0.1

    def __init__(self, transformer_name: str, n_extra_features: int = 0) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(transformer_name, num_labels=self.N_LABELS)
        config.gradient_checkpointing = True
        self._encoder = AutoModel.from_pretrained(transformer_name, config=config)
        self._loss_function = nn.CrossEntropyLoss()
        self._classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_size + n_extra_features, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.DROPOUT_RATE),
            nn.Linear(config.hidden_size, self.N_LABELS)
        )

    @autocast()
    def forward(
            self,
            tokens: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            indexes_1: Optional[torch.Tensor] = None,
            indexes_2: Optional[torch.Tensor] = None,
            features: Optional[torch.Tensor] = None
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

        if features is not None:
            h = torch.cat((h, features), dim=-1)

        logits = self._classifier(h)
        outputs = (logits,)

        if labels is not None:
            loss = self._loss_function(logits.float(), labels)
            outputs = (loss,) + outputs

        return outputs
