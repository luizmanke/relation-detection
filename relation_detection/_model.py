import numpy as np
from datetime import datetime as dt
from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit
from typing import Any, Dict, List, Tuple
from . import Dataset
from .models.between import Between
from .models.catboost import CatBoost
from .models.surround import Surround
from .models.transformer import Transformer


class Model:

    available_methods_ = {
        "between": Between,
        "catboost": CatBoost,
        "surround": Surround,
        "transformer": Transformer
    }

    def __init__(self) -> None:
        pass

    def list_available_methods(self) -> List[str]:
        return list(self.available_methods_.keys())

    def load(self, model_name: str) -> None:
        assert model_name in self.available_methods_
        self.model_name_ = model_name

    def cross_validate(self, dataset: Dataset) -> None:
        self.results_: Dict[str, Any] = {"train": {}, "test": {}}
        self.samples_, self.labels_, self.groups_ = dataset.get_data()
        for fold, (indexes_train, indexes_test) in enumerate(self._create_splits()):
            self._train(indexes_train, fold)
            self._test(indexes_test, fold)

    def _create_splits(self) -> Any:
        gss = GroupShuffleSplit(train_size=0.8, n_splits=5, random_state=42)
        return gss.split(self.samples_, self.labels_, self.groups_)

    def _train(self, indexes: np.ndarray, fold: int) -> None:
        samples_train, labels_train = self._select_samples(self.samples_, self.labels_, indexes)

        start_time = dt.now()
        self.model_ = self.available_methods_[self.model_name_]()
        self.model_.fit(samples_train, labels_train)
        elapsed_time = dt.now() - start_time

        labels_pred_train = self.model_.predict(samples_train)
        self._evaluate(labels_train, labels_pred_train, fold, "train")
        self.results_["train"][fold]["time"] = elapsed_time

    def _test(self, indexes: np.ndarray, fold: int) -> None:
        samples_test, labels_test = self._select_samples(self.samples_, self.labels_, indexes)
        labels_pred_test = self.model_.predict(samples_test)
        self._evaluate(labels_test, labels_pred_test, fold, "test")

    @staticmethod
    def _select_samples(
            samples: List[dict], labels: np.ndarray, indexes: np.ndarray
    ) -> Tuple[List[dict], np.ndarray]:
        selected_samples = [samples[i] for i in indexes]
        selected_labels = labels[indexes]
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
