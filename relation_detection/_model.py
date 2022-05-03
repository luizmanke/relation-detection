import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime as dt
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from .models.graph import Graph
from .models.surround import Surround
from .models.transformer import Transformer
from .utils import print_sentence


class Model:

    n_folds_ = 5
    available_models_ = {
        "graph": Graph,
        "surround": Surround,
        "transformer": Transformer
    }

    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name_ = model_name
        self.kwargs_ = kwargs

    def train(self, dataset: Any) -> None:
        self._train_setup(dataset)
        indexes_train, indexes_test = next(self._create_splits())
        self._train(indexes_train, 0)
        self._test(indexes_test, 0)

    def predict(
            self,
            samples: List[dict],
            return_proba: bool = False,
            for_explainer: bool = False
    ) -> np.ndarray:
        predictions, predictions_proba = self.model_.predict(samples, for_explainer)
        if return_proba:
            return predictions_proba
        else:
            return predictions

    def cross_validate(self, dataset: Any) -> None:
        self._train_setup(dataset)
        for fold, (indexes_train, indexes_test) in tqdm(
                enumerate(self._create_splits()), total=self.n_folds_):
            self._train(indexes_train, fold)
            self._test(indexes_test, fold)

    def get_indexes_of(self, label: str, order: str = "desc") -> np.ndarray:
        condition = self._get_condition(label)
        return self._get_indexes(condition, order)

    def get_results(self) -> pd.DataFrame:
        results: dict = {}

        if hasattr(self, "results_"):
            for split, first_level in self.results_.items():
                for fold, second_level in first_level.items():
                    for metric, score in second_level.items():
                        key = (split, metric)
                        if key not in results:
                            results[key] = []
                        results[key].append(score)

        df = pd.DataFrame(results)
        df.index.name = "fold"
        return df.transpose()

    def plot_confusion_matrix(self) -> None:
        _, ax = plt.subplots(figsize=(10, 6))
        ConfusionMatrixDisplay.from_predictions(
            self.predictions_,
            self.data_["labels"],
            normalize="true",
            ax=ax
        )
        ax.set_xlabel("Predicted label", fontsize=16, labelpad=16)
        ax.set_ylabel("True label", fontsize=16, labelpad=16)

    def plot_kde(self) -> None:
        _, ax = plt.subplots(figsize=(16, 6))
        sns.kdeplot(
            data=pd.DataFrame({"label": self.data_["labels"], "proba": self.predictions_proba_[:, 1]}),
            x="proba",
            hue="label",
            ax=ax
        )
        ax.set_xlabel("Probability", fontsize=16, labelpad=16)
        ax.set_ylabel("Density", fontsize=16, labelpad=16)
        ax.grid(True)

    def print_sentence(self, index: int) -> None:
        sample = {key: value for key, value in self.data_["samples"][index].items()}
        print("true label:", self.data_["labels"][index])
        print("prediction:", self.predictions_proba_[index][1])
        print_sentence(sample)

    def _train_setup(self, dataset: Any) -> None:
        self.results_: Dict[str, Any] = {"train": {}, "test": {}}
        self.data_ = dataset.get_data()
        self.predictions_ = np.empty(len(self.data_["samples"]))
        self.predictions_proba_ = np.empty((len(self.data_["samples"]), 2))

    def _create_splits(self) -> Any:
        gss = GroupKFold(n_splits=self.n_folds_)
        return gss.split(self.data_["samples"], self.data_["labels"], self.data_["groups"])

    def _train(self, indexes: np.ndarray, fold: int) -> None:
        samples_train, labels_train = self._select_samples(indexes)

        start_time = dt.now()
        self.model_ = self.available_models_[self.model_name_](**self.kwargs_)
        self.model_.fit(samples_train, labels_train)
        elapsed_time = dt.now() - start_time

        labels_pred_train, _ = self.model_.predict(samples_train)
        self._evaluate(labels_train, labels_pred_train, fold, "train")
        self.results_["train"][fold]["time"] = elapsed_time

    def _test(self, indexes: np.ndarray, fold: int) -> None:
        samples_test, labels_test = self._select_samples(indexes)

        start_time = dt.now()
        labels_pred_test, labels_proba_test = self.model_.predict(samples_test)
        elapsed_time = dt.now() - start_time

        self._evaluate(labels_test, labels_pred_test, fold, "test")
        self.results_["test"][fold]["time"] = elapsed_time
        self.predictions_[indexes] = labels_pred_test
        self.predictions_proba_[indexes] = labels_proba_test

    def _select_samples(self, indexes: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        selected_samples = [self.data_["samples"][i] for i in indexes]
        selected_labels = self.data_["labels"][indexes]
        return selected_samples, selected_labels

    def _evaluate(
            self, labels_true: np.ndarray, labels_pred: np.ndarray, fold: int, split: str
    ) -> None:
        tn, fp, fn, tp = metrics.confusion_matrix(labels_true, labels_pred).ravel()
        self.results_[split][fold] = {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "accuracy": round(metrics.accuracy_score(labels_true, labels_pred), 4),
            "recall": round(metrics.recall_score(labels_true, labels_pred), 4),
            "precision": round(metrics.precision_score(labels_true, labels_pred), 4),
            "f1": round(metrics.f1_score(labels_true, labels_pred), 4),
            "mcc": round(metrics.matthews_corrcoef(labels_true, labels_pred), 4)
        }

    def _get_condition(self, label: str) -> np.ndarray:
        assert label in ["TP", "TN", "FP", "FN"]

        if label[0] == "T":
            condition_1 = self.predictions_ == self.data_["labels"]
        elif label[0] == "F":
            condition_1 = self.predictions_ != self.data_["labels"]

        if label[1] == "P":
            condition_2 = self.predictions_ == 1
        elif label[1] == "N":
            condition_2 = self.predictions_ == 0

        return condition_1 & condition_2

    def _get_indexes(self, condition: np.ndarray, order: str) -> np.ndarray:
        indexes = np.arange(len(self.predictions_))
        indexes_selected = indexes[condition]
        predictions_proba_selected = self.predictions_proba_[condition, 1]
        sorted_args_selected = predictions_proba_selected.argsort()
        indexes_selected = indexes_selected[sorted_args_selected]
        indexes_selected = indexes_selected[::-1] if order == "desc" else indexes_selected
        return indexes_selected
