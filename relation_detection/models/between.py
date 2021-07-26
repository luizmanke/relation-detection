import numpy as np
from typing import List
from .base.classifier import BaseClassifier
from .base.vectorizer import BaseVectorizer


class Between(BaseVectorizer, BaseClassifier):

    def __init__(self, **kwargs):
        BaseVectorizer.__init__(self, vectorizer_name="count")
        BaseClassifier.__init__(self, **kwargs)

    def fit(self, samples: List[dict], y: np.ndarray) -> None:  # type: ignore[override]
        sentences = self._get_middle_sentences(samples)
        self._vectorizer_fit(sentences)
        x = self._vectorizer_transform(sentences)
        BaseClassifier.fit(self, x, y)

    def predict(self, samples: List[dict]) -> np.ndarray:  # type: ignore[override]
        sentences = self._get_middle_sentences(samples)
        x = self._vectorizer_transform(sentences)
        return BaseClassifier.predict(self, x)

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
