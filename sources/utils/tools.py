import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split as tt_split
from typing import Any, Tuple


def train_test_split(x: Any, y: np.ndarray) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    return tt_split(x, y, stratify=y, train_size=0.8, random_state=42)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    print(f'TP: {tp}')
    print(f'TN: {tn}')
    print(f'FP: {fp}')
    print(f'FN: {fn}')
    print(f"{'accuracy:':{10}} {metrics.accuracy_score(y_true, y_pred):.3}")
    print(f"{'recall:':{10}} {metrics.recall_score(y_true, y_pred):.3}")
    print(f"{'precision:':{10}} {metrics.precision_score(y_true, y_pred):.3}")
    print(f"{'f1:':{10}} {metrics.f1_score(y_true, y_pred):.3}")
    print(f"{'mcc:':{10}} {metrics.matthews_corrcoef(y_true, y_pred):.3}")
