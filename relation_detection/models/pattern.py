import numpy as np
from sklearn.metrics import f1_score
from typing import List, Tuple
from .base.engineering import BaseEngineering
from .base.metaheuristic import BaseMetaheuristic


class Pattern(BaseEngineering, BaseMetaheuristic):

    FEATURES = {
        "n_tokens_between_entities": [0, 50],
        "n_commas_between_entities": [0, 10],
        "n_semicolons_between_entities": [0, 10],
        "n_colons_between_entities": [0, 10],
        "n_periods_between_entities": [0, 10],
        "n_exclamation_points_between_entities": [0, 10],
        "n_question_marks_between_entities": [0, 10],
        "n_quotation_marks_between_entities": [0, 10],
        "n_stopwords_between_entities": [0, 20],
        "n_prop_names_between_entities": [0, 20],
        "n_pronouns_between_entities": [0, 20],
        "n_neg_adverbs_between_entities": [0, 20]
    }

    def __init__(self, **kwargs):
        BaseEngineering.__init__(self)
        BaseMetaheuristic.__init__(
            self,
            method_name="harmony_search",
            objective_function=self._objective_function,
            features=self.FEATURES
        )
        self.patterns_ = np.array([])

    def fit(  # type: ignore[override]
            self,
            samples: List[dict],
            y: np.ndarray,
            groups: List[str]
    ) -> None:
        x = self._compute_features(samples)
        patterns_per_run, self.history_ = BaseMetaheuristic.get_best_population(self, x, y)
        self.patterns_ = self._select_best_patterns(x, y, patterns_per_run)

    def predict(  # type: ignore[override]
            self,
            samples: List[dict],
            for_explainer: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = self._compute_features(samples)
        predictions_proba = self._predict_proba(x, self.patterns_)
        predictions = self._proba_to_pred(predictions_proba)
        return predictions, predictions_proba

    @staticmethod
    def _predict_proba(x: np.ndarray, patterns: List[list]) -> np.ndarray:
        predictions_count = np.zeros((len(x), len(patterns)))
        for i, pattern in enumerate(patterns):
            predictions_count[:, i] = np.where((x == pattern).all(axis=1), 1, 0)
        return np.array([
            (predictions_count == 0).sum(axis=1) / len(patterns),
            (predictions_count == 1).sum(axis=1) / len(patterns)
        ]).transpose()

    @staticmethod
    def _proba_to_pred(predictions_proba: np.ndarray,) -> np.ndarray:
        return np.where(predictions_proba[:, 1] > 0, 1, 0)

    def _compute_features(self, samples: List[dict]) -> np.ndarray:
        features = np.zeros((len(samples), len(self.FEATURES)))
        for i, feature in enumerate(self.FEATURES):
            method = getattr(self, f"_compute_{feature}")
            features[:, i] = [method(sample) for sample in samples]
        return features

    @staticmethod
    def _objective_function(x: np.ndarray, y: np.ndarray, pattern: list) -> float:
        y_pred = np.where((x == pattern).all(axis=1), 1, 0)
        return f1_score(y, y_pred)

    def _select_best_patterns(
            self,
            x: np.ndarray,
            y: np.ndarray,
            patterns_per_run: np.ndarray
    ) -> np.ndarray:
        best_patterns = np.array([])
        best_score = float("-inf")
        for run in range(patterns_per_run.shape[0]):
            current_patterns = patterns_per_run[run, :, :]
            predictions_proba = self._predict_proba(x, current_patterns)
            predictions = self._proba_to_pred(predictions_proba)
            current_score = f1_score(y, predictions)
            if current_score > best_score:
                best_patterns = current_patterns
                best_score = current_score
        return best_patterns
