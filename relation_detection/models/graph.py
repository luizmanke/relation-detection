import numpy as np
import random
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .base import tree
from .base.nlp import NLP

# Globals
PAD_ID = 0
SEED = 42

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class Graph(NLP):

    BATCH_SIZE = 50
    DECAY_MIN_EPOCHS = 5
    DECAY_LEARNING_RATE = 0.98
    GRADIENT_MAX_NORM = 5
    LEARNING_RATE = 1
    L2_REGULARIZATION = 0
    N_EPOCHS = 100
    POOLING_L2 = 0.003

    def __init__(self):
        NLP.__init__(self)

    def fit(  # type: ignore[override]
        self,
        samples: List[dict],
        y: np.ndarray
    ) -> None:

        NLP.fit(self, samples)
        inputs = NLP.extract(self, samples, word_dropout=True)
        data_loader = self._create_data_loader(inputs, y, shuffle=True)
        device = self._get_device()
        self._create_model(device)
        self._set_up_optimizers(device)
        self._fit(data_loader, device)

    def predict(  # type: ignore[override]
        self,
        samples: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:

        inputs = NLP.extract(self, samples, word_dropout=False)
        data_loader = self._create_data_loader(inputs, y=None, shuffle=False)
        device = self._get_device()
        predictions_proba = self._predict_proba(data_loader, device)
        predictions = predictions_proba.argmax(axis=1)
        return predictions, predictions_proba

    def _predict_proba(self, data_loader: DataLoader, device: torch.device) -> np.ndarray:
        predictions_proba = []
        for batch in data_loader:

            inputs = {}
            for key, value in batch.items():
                inputs[key] = value.to(device) if key != "indexes" else value

            self.model_.eval()
            outputs = self.model_(inputs)
            predictions_proba += F.softmax(outputs["logits"], dim=1).data.cpu().numpy().tolist()

        return np.array(predictions_proba)

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

    def _collate_fn(self, batch: list) -> dict:

        batch_size = len(batch)

        def to_long_tensor(some_list: list):
            max_len = max(len(items) for items in some_list)
            long_tensor = torch.LongTensor(batch_size, max_len).fill_(PAD_ID)
            for i, items in enumerate(some_list):
                long_tensor[i, :len(items)] = torch.LongTensor(items)
            return long_tensor

        # sort to ease RNN operations
        indexes = [i for i in range(len(batch))]
        sorted_zip = sorted(zip(batch, indexes), key=lambda x: len(x[0]["tokens"]), reverse=True)
        batch_sorted, indexes_sorted = zip(*sorted_zip)

        # list of dicts to dict of lists
        batch_refactored = {k: [d[k] for d in batch_sorted] for k in batch_sorted[0]}

        return {
            "tokens": to_long_tensor(batch_refactored["tokens"]),
            "masks": torch.eq(to_long_tensor(batch_refactored["tokens"]), 0),
            "heads": to_long_tensor(batch_refactored["heads"]),
            "dependencies": to_long_tensor(batch_refactored["dependencies"]),
            "part_of_speeches": to_long_tensor(batch_refactored["part_of_speeches"]),
            "types": to_long_tensor(batch_refactored["types"]),
            "positions_1": to_long_tensor(batch_refactored["position_1"]),
            "positions_2": to_long_tensor(batch_refactored["position_2"]),
            "labels": torch.LongTensor(batch_refactored["label"]) if "label" in batch_refactored else None,
            "indexes": indexes_sorted
        }

    @staticmethod
    def _get_device() -> torch.device:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        return device

    def _create_model(self, device: torch.device) -> None:
        self.model_ = BaseCNN(self.vectors_, self.maps_, device)
        self.model_.to(device=device)

    def _set_up_optimizers(self, device: torch.device) -> None:

        self._loss_function = nn.CrossEntropyLoss().to(device)

        parameters = [p for p in self.model_.parameters() if p.requires_grad]
        self._optimizer = torch.optim.SGD(
            parameters,
            lr=self.LEARNING_RATE,
            weight_decay=self.L2_REGULARIZATION
        )

    def _fit(self, data_loader: DataLoader, device: torch.device) -> None:

        for epoch in range(self.N_EPOCHS):
            for batch in data_loader:

                # prepare model
                self.model_.train()
                self._optimizer.zero_grad()

                # train model
                inputs = {}
                for key, value in batch.items():
                    inputs[key] = value.to(device) if key != "indexes" else value
                outputs = self.model_(inputs)

                # compute loss
                print(outputs["logits"].get_device(), batch["labels"].get_device())
                loss = self._loss_function(outputs["logits"], batch["labels"])
                loss += self.POOLING_L2 * (outputs["pooling"] ** 2).sum(1).mean()

                # optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.GRADIENT_MAX_NORM)
                self._optimizer.step()

                # update learning rate
                if epoch > self.DECAY_MIN_EPOCHS:
                    new_lerning_rate = self.LEARNING_RATE * (epoch ** self.DECAY_LEARNING_RATE)
                    for param_group in self._optimizer.param_groups:
                        param_group["lr"] = new_lerning_rate


class BaseCNN(nn.Module):

    CLASSIFIER_HIDDEN_SIZE = 200
    CLASSIFIER_N_CLASSES = 2
    CNN_HIDDEN_SIZE = 200
    CNN_N_LAYERS = 2
    DROPOUT = 0.5
    EMBEDDINGS_POS_SIZE = 30
    EMBEDDINGS_NER_SIZE = 30
    MLP_HIDDEN_SIZE = 200
    MLP_N_LAYERS = 2
    RNN_BIDIRECTIONAL = True
    RNN_HIDDEN_SIZE = 200
    RNN_N_LAYERS = 1

    def __init__(self, embeddings: np.ndarray, maps: dict, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self._tree = Tree(device)
        self._embeddings = Embeddings(
            embeddings,
            maps,
            self.EMBEDDINGS_POS_SIZE,
            self.EMBEDDINGS_NER_SIZE,
            self.DROPOUT
        )
        self._rnn = RNN(
            embeddings.shape[1] + self.EMBEDDINGS_POS_SIZE + self.EMBEDDINGS_NER_SIZE,
            self.RNN_HIDDEN_SIZE,
            self.RNN_N_LAYERS,
            self.RNN_BIDIRECTIONAL,
            self.DROPOUT
        )
        self._cnn = CNN(
            self.RNN_HIDDEN_SIZE * 2,
            self.CNN_HIDDEN_SIZE,
            self.CNN_N_LAYERS,
            self.DROPOUT
        )
        self._mlp = MLP(
            self.MLP_HIDDEN_SIZE * 3,
            self.MLP_HIDDEN_SIZE,
            self.MLP_N_LAYERS
        )
        self._classifier = Classifier(self.CLASSIFIER_HIDDEN_SIZE, self.CLASSIFIER_N_CLASSES)

    def forward(self, inputs: dict) -> dict:
        tree_outputs = self._tree.get_matrix(inputs)
        embeddings_outputs = self._embeddings(inputs)
        rnn_outputs = self._rnn({**inputs, **embeddings_outputs}, self._device)
        cnn_outputs = self._cnn({**inputs, **tree_outputs, **rnn_outputs})
        mlp_outputs = self._mlp({**inputs, **cnn_outputs})
        classifier_outputs = self._classifier(mlp_outputs)
        return classifier_outputs


class Tree:

    def __init__(self, device: torch.device) -> None:
        self._device = device

    def get_matrix(self, inputs: dict) -> dict:
        heads = inputs["heads"].cpu().numpy()
        lengths = (inputs["masks"].data.cpu().numpy() == 0).astype(np.int64).sum(1)
        max_length = max(lengths)

        trees = [tree.head_to_tree(heads[i], lengths[i]) for i in range(len(lengths))]
        matrix = [tree.tree_to_matrix(item, max_length) for item in trees]

        return {"tree_matrix": Variable(torch.from_numpy(np.concatenate(matrix, axis=0))).to(self._device)}


class Embeddings(nn.Module):

    def __init__(
        self,
        embeddings: np.ndarray,
        maps: dict,
        part_of_speeches_size: int,
        types_size: int,
        dropout: float
    ) -> None:

        super().__init__()

        # words
        vocab_size, vocab_dim = embeddings.shape
        self._word_embeddings = nn.Embedding(vocab_size, vocab_dim, padding_idx=PAD_ID)
        self._word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))

        # part of speeches
        self._part_of_speech_embeddings = nn.Embedding(len(maps["part_of_speeches"]), part_of_speeches_size)

        # types
        self._types_embeddings = nn.Embedding(len(maps["types"]), types_size)

        # dropout
        self._dropout = nn.Dropout(dropout)

    def forward(self, inputs: dict) -> dict:
        embeddings = []
        embeddings.append(self._word_embeddings(inputs["tokens"]))
        embeddings.append(self._part_of_speech_embeddings(inputs["part_of_speeches"]))
        embeddings.append(self._types_embeddings(inputs["types"]))
        embeddings_tensor = torch.cat(embeddings, dim=2)
        return {"embeddings": self._dropout(embeddings_tensor)}


class RNN(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        bidirectional: bool,
        dropout: float
    ) -> None:

        super().__init__()
        self._hidden_size = hidden_size
        self._n_hidden_layers = n_hidden_layers
        self._bidirectional = bidirectional
        self._lstm = nn.LSTM(
            input_size,
            hidden_size,
            n_hidden_layers,
            dropout=dropout,
            bidirectional=self._bidirectional,
            batch_first=True
        )
        self._dropout = nn.Dropout(dropout)

    def forward(self, inputs: dict, device: torch.device) -> dict:

        # get rnn zero state
        batch_size = inputs["tokens"].size()[0]
        total_layers = self._n_hidden_layers * 2 if self._bidirectional else self._n_hidden_layers
        state_shape = (total_layers, batch_size, self._hidden_size)
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False).to(device)

        # rnn layers
        lengths = list(inputs["masks"].data.eq(PAD_ID).long().sum(1).squeeze())
        padded_sequence = nn.utils.rnn.pack_padded_sequence(
            inputs["embeddings"],
            lengths,  # type: ignore
            batch_first=True
        )
        lstm_output, (_, _) = self._lstm(padded_sequence, (h0, c0))
        packed_sequence, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        return {"rnn": self._dropout(packed_sequence)}


class CNN(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        dropout: float
    ) -> None:

        super().__init__()
        self._n_hidden_layers = n_hidden_layers

        self.W = nn.ModuleList()
        for layer in range(n_hidden_layers):
            input_dim = input_size if layer == 0 else hidden_size
            self.W.append(nn.Linear(input_dim, hidden_size))
        self._dropout = nn.Dropout(dropout)

    def forward(self, inputs: dict) -> dict:

        denom = inputs["tree_matrix"].sum(2).unsqueeze(2) + 1
        tree_masks = (inputs["tree_matrix"].sum(2) + inputs["tree_matrix"].sum(1)).eq(0).unsqueeze(2)
        cnn_outputs = inputs["rnn"]
        for layer in range(self._n_hidden_layers):
            Ax = inputs["tree_matrix"].bmm(cnn_outputs)
            AxW = self.W[layer](Ax)
            AxW = AxW + self.W[layer](cnn_outputs)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cnn_outputs = self._dropout(gAxW) if layer < self._n_hidden_layers - 1 else gAxW

        return {"cnn": cnn_outputs, "tree_masks": tree_masks}


class MLP(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_hidden_layers: int
    ) -> None:

        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self._mlp = nn.Sequential(*layers)

    def forward(self, inputs: dict) -> dict:
        tree_masks = self._pool(inputs["cnn"], inputs["tree_masks"])
        positions_1 = self._pool(inputs["cnn"], inputs["positions_1"].eq(0).eq(0).unsqueeze(2))
        positions_2 = self._pool(inputs["cnn"], inputs["positions_2"].eq(0).eq(0).unsqueeze(2))
        values_concat = torch.cat([tree_masks, positions_1, positions_2], dim=1)
        return {"mlp": self._mlp(values_concat), "pooling": tree_masks}

    @staticmethod
    def _pool(value, mask):
        value = value.masked_fill(mask, 0)
        return value.sum(1)


class Classifier(nn.Module):

    def __init__(self, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        self._classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, inputs: dict) -> dict:
        return {"logits": self._classifier(inputs["mlp"]), "pooling": inputs["pooling"]}
