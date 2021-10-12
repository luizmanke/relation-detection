import numpy as np
from sklearn.metrics import f1_score
from typing import List, Tuple
from .base.engineering import BaseEngineering
from .base.metaheuristic import BaseMetaheuristic


class Pattern(BaseEngineering, BaseMetaheuristic):

    def __init__(self, **kwargs):
        BaseEngineering.__init__(self, use_numerical=False, use_categorical=True)
        BaseMetaheuristic.__init__(
            self,
            method_name="harmony_search",
            objective_function=self._objective_function
        )
        self.patterns_ = []
        self.history_ = []

    def fit(  # type: ignore[override]
            self,
            samples: List[dict],
            y: np.ndarray,
            groups: List[str]
    ) -> None:
        BaseEngineering.fit(self, samples)
        x = BaseEngineering.transform(self, samples).values
        self.patterns_, self.history_ = BaseMetaheuristic.fit_and_return(self, x, y)

    def predict(  # type: ignore[override]
            self,
            samples: List[dict],
            for_explainer: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = BaseEngineering.transform(self, samples).values
        predictions_proba = self._predict_proba(x, self.patterns_)
        predictions = self._proba_to_pred(predictions_proba)
        return predictions, predictions_proba

    @staticmethod
    def _predict_proba(x: np.ndarray, patterns: List[list]) -> np.ndarray:
        predictions_count = np.zeros((len(x), len(patterns)))
        for i, pattern in enumerate(patterns):
            x_tmp = x.copy()
            for j, pattern_number in enumerate(pattern):
                if pattern_number == 1:  # ANY
                    x_tmp[:, j] = pattern_number
            predictions_count[:, i] = np.where((x_tmp == pattern).all(axis=1), 1, 0)
        return np.array([
            (predictions_count == 0).sum(axis=1) / len(patterns),
            (predictions_count == 1).sum(axis=1) / len(patterns)
        ]).transpose()

    @staticmethod
    def _proba_to_pred(predictions_proba: np.ndarray,) -> np.ndarray:
        return np.where(predictions_proba[:, 1] > 0, 1, 0)

    def _objective_function(self, x: np.ndarray, y: np.ndarray, patterns: List[list]) -> float:
        predictions_proba = self._predict_proba(x, patterns)
        predictions = self._proba_to_pred(predictions_proba)
        return f1_score(y, predictions)
