import numpy as np
from .base_classifier import BaseClassifier
from sklearn.feature_extraction.text import CountVectorizer
from typing import List


class CountVector(BaseClassifier):

    def __init__(self, **kwargs):
        BaseClassifier.__init__(self, **kwargs)

    def fit(self, samples: List[dict], y: np.ndarray) -> None:
        sentences = self._get_middle_sentences(samples)
        self._vectorizer_fit(sentences)
        x = self._vectorizer_transform(sentences)
        BaseClassifier.fit(self, x, y)

    def predict(self, samples: List[dict]) -> np.ndarray:
        sentences = self._get_middle_sentences(samples)
        x = self._vectorizer_transform(sentences)
        return BaseClassifier.predict(self, x)

    @staticmethod
    def _get_middle_sentences(samples: List[dict]) -> List[str]:
        middle_sentences = []
        for sample in samples:
            middle_tokens = sample["tokens"][sample['index_1']+1:sample['index_2']]
            middle_sentences.append(" ".join(middle_tokens))
        return middle_sentences

    def _vectorizer_fit(self, sentences: List[str]) -> None:
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(sentences)

    def _vectorizer_transform(self, sentences: List[str]):
        return self.vectorizer.transform(sentences).toarray()
