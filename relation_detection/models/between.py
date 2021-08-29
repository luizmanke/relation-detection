import numpy as np
from typing import List, Tuple
from .base.classifier import BaseClassifier
from .base.vectorizer import BaseVectorizer


class Between(BaseVectorizer, BaseClassifier):

    def __init__(self, **kwargs):
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
        BaseClassifier.fit(self, x, y)

    def predict(  # type: ignore[override]
            self,
            samples: List[dict],
            for_explainer: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        sentences = self._get_middle_sentences(samples)
        x = self._vectorizer_transform(sentences)
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
