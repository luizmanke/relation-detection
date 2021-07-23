import argparse
import json
import numpy as np
import os
import pickle
from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit
from typing import Any, Dict, List, Tuple


# Globals
DATASETS: Dict[str, Any] = {}
MODELS: Dict[str, Any] = {}
RESULTS_DIR = "results"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="dbpedia")
    parser.add_argument("--model_name", type=str, default="count_vector")
    parser.add_argument("--quiet", type=bool, default=False)
    args = parser.parse_args()

    _load_datasets(args.dataset_name)
    _load_models(args.model_name)

    return args


def _load_datasets(dataset_name: str) -> None:
    is_all = True if dataset_name == "all" else False
    if dataset_name == "dbpedia" or is_all:
        from .preprocess.dbpedia import DBpedia
        DATASETS["dbpedia"] = DBpedia("data/DBpediaRelations-PT-0.2.txt")


def _load_models(model_name: str) -> None:
    compute_all = True if model_name == "all" else False
    if model_name == "transformer" or compute_all:
        from .modeling.transformer import Transformer
        MODELS["transformer"] = Transformer
    if model_name == "catboost" or compute_all:
        from .modeling.catboost import CatBoost
        MODELS["catboost"] = CatBoost
    if model_name == "between" or compute_all:
        from .modeling.between import Between
        MODELS["between"] = Between
    if model_name == "surround" or compute_all:
        from .modeling.surround import Surround
        MODELS["surround"] = Surround


def train_test_split(x: List[dict], y: np.ndarray) -> Tuple[
        List[dict], List[dict], np.ndarray, np.ndarray]:

    groups = [" ".join(item["tokens"]) for item in x]
    indexes_train, indexes_test = next(
        GroupShuffleSplit(train_size=0.8, n_splits=2, random_state=42)
        .split(x, y, groups)
    )
    x_train = [x[i] for i in indexes_train]
    x_test = [x[i] for i in indexes_test]
    y_train = y[indexes_train]
    y_test = y[indexes_test]

    return x_train, x_test, y_train, y_test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": round(metrics.accuracy_score(y_true, y_pred), 4),
        "recall": round(metrics.recall_score(y_true, y_pred), 4),
        "precision": round(metrics.precision_score(y_true, y_pred), 4),
        "f1": round(metrics.f1_score(y_true, y_pred), 4),
        "mcc": round(metrics.matthews_corrcoef(y_true, y_pred), 4)
    }


def save_model(dataset_name: str, model_name: str) -> None:
    dir = f"{RESULTS_DIR}/{dataset_name}/{model_name}"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    with open(f"{dir}/model.pickle", "wb") as file:
        pickle.dump(MODELS[model_name], file)


def load_model(dataset_name: str, model_name: str):
    dir = f"{RESULTS_DIR}/{dataset_name}/{model_name}"
    with open(f"{dir}/model.pickle", "rb") as file:
        model = pickle.load(file)
    return model


def save_scores(
        scores: dict,
        dir: str,
        source: str
) -> None:
    if not os.path.isdir(dir):
        os.makedirs(dir)
    with open(f"{dir}/scores_{source}.json", "w") as file:
        json.dump(scores, file)
