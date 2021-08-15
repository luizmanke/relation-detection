import numpy as np
from typing import List, Tuple
from .base.classifier import BaseClassifier
from .base.engineering import BaseEngineering
from .base.vectorizer import BaseVectorizer


class Between(BaseEngineering, BaseVectorizer, BaseClassifier):

    def __init__(self, **kwargs):
        self.add_features_ = False
        BaseEngineering.__init__(self)
        BaseVectorizer.__init__(self, vectorizer_name="count")
        BaseClassifier.__init__(self, **kwargs)

    def fit(  # type: ignore[override]
            self,
            samples: List[dict],
            y: np.ndarray,
            groups: List[str]
    ) -> None:
        sentences = self._get_middle_sentences(samples)
        self._vectorizer_fit(sentences)
        x = self._vectorizer_transform(sentences)
        x = self._add_features(samples, x, fit=True) if self.add_features_ else x
        BaseClassifier.fit(self, x, y)

    def predict(  # type: ignore[override]
            self,
            samples: List[dict],
            for_lime: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        sentences = self._get_middle_sentences(samples)
        x = self._vectorizer_transform(sentences)
        x = self._add_features(samples, x) if self.add_features_ else x
        predictions_proba = BaseClassifier.predict(self, x)
        predictions = predictions_proba.argmax(axis=1)
        return predictions, predictions_proba

    @staticmethod
    def _get_middle_sentences(samples: List[dict]) -> List[str]:
        middle_sentences = []
        for sample in samples:
            middle_tokens = sample["tokens"][sample["index_1"]+1:sample["index_2"]]
            middle_sentences.append(" ".join(middle_tokens))
        return middle_sentences

    def _vectorizer_fit(self, sentences: List[str]) -> None:
        BaseVectorizer.fit(self, sentences)

    def _vectorizer_transform(self, sentences: List[str]) -> np.ndarray:
        return BaseVectorizer.transform(self, sentences)

    def _add_features(
            self,
            samples: List[dict],
            x: np.ndarray,
            fit: bool = False
    ) -> np.ndarray:
        if fit:
            BaseEngineering.fit(self, samples)
        x_features = BaseEngineering.get_features(self, samples).to_numpy()
        return np.concatenate([x, x_features], axis=1)
