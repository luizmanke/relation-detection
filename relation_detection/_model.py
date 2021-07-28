import numpy as np
from datetime import datetime as dt
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from . import Dataset
from .models.between import Between
from .models.catboost import CatBoost
from .models.surround import Surround
from .models.transformer import Transformer


class Model:

    n_folds_ = 5
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
        self.predictions_ = np.zeros(len(self.samples_))
        for fold, (indexes_train, indexes_test) in tqdm(
                enumerate(self._create_splits()), total=self.n_folds_):
            self._train(indexes_train, fold)
            self._test(indexes_test, fold)

    def _create_splits(self) -> Any:
        gss = GroupKFold(n_splits=self.n_folds_)
        return gss.split(self.samples_, self.labels_, self.groups_)

    def _train(self, indexes: np.ndarray, fold: int) -> None:
        samples_train, labels_train, groups_train = self._select_samples(indexes)

        start_time = dt.now()
        self.model_ = self.available_methods_[self.model_name_]()
        self.model_.fit(samples_train, labels_train, groups_train)
        elapsed_time = dt.now() - start_time

        labels_pred_train = self.model_.predict(samples_train)
        self._evaluate(labels_train, labels_pred_train, fold, "train")
        self.results_["train"][fold]["time"] = elapsed_time

    def _test(self, indexes: np.ndarray, fold: int) -> None:
        samples_test, labels_test, _ = self._select_samples(indexes)

        start_time = dt.now()
        labels_pred_test = self.model_.predict(samples_test)
        elapsed_time = dt.now() - start_time

        self._evaluate(labels_test, labels_pred_test, fold, "test")
        self.results_["test"][fold]["time"] = elapsed_time
        self.predictions_[indexes] = labels_pred_test

    def _select_samples(self, indexes: np.ndarray) -> Tuple[List[dict], np.ndarray, List[str]]:
        selected_samples = [self.samples_[i] for i in indexes]
        selected_labels = self.labels_[indexes]
        selected_groups = [self.groups_[i] for i in indexes]
        return selected_samples, selected_labels, selected_groups

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
