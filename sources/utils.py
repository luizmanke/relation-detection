import argparse
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split as tt_split
from typing import Any, Tuple
from .modeling.count_vector import CountVector
from .preprocess.dbpedia import DBpedia


DATASETS = {
    "dbpedia": DBpedia("data/DBpediaRelations-PT-0.2.txt")
}
MODELS = {
    "count_vector": CountVector()
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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return pd.DataFrame([{
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
        "mcc": metrics.matthews_corrcoef(y_true, y_pred)
    }])


def save_model(model_name: str) -> None:
    dir = f"{RESULTS_DIR}/{model_name}"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    with open(f"{dir}/model.pickle", "wb") as file:
        pickle.dump(MODELS[model_name], file)


def load_model(model_name: str):
    with open(f"{RESULTS_DIR}/{model_name}/model.pickle", "rb") as file:
        model = pickle.load(file)
    return model


def save_scores(df_scores: pd.DataFrame, model_name: str, source: str) -> None:
    dir = f"{RESULTS_DIR}/{model_name}"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    df_scores.to_csv(f"{dir}/scores_{source}.csv")
