import argparse
import json
import numpy as np
import os
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split as tt_split
from typing import Any, Tuple
from .modeling.bert import BERT
from .modeling.count_vector import CountVector
from .modeling.prefix_middle_suffix import PrefixMiddleSuffix
from .preprocess.dbpedia import DBpedia


DATASETS = {
    "dbpedia": DBpedia("data/DBpediaRelations-PT-0.2.txt")
}
MODELS = {
    "bert": BERT,
    "count_vector": CountVector,
    "prefix_middle_suffix": PrefixMiddleSuffix
}
RESULTS_DIR = "results"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="dbpedia")
    parser.add_argument("--model_name", type=str, default="count_vector")
    parser.add_argument("--quiet", type=bool, default=False)
    args = parser.parse_args()

    if not args.quiet:
        print("\n## Input args:")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")

    return args


def train_test_split(x: Any, y: np.ndarray) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    return tt_split(x, y, stratify=y, train_size=0.8, random_state=42)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
        "mcc": metrics.matthews_corrcoef(y_true, y_pred)
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
