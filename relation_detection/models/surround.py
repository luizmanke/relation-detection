import numpy as np
from typing import List, Tuple
from .base.classifier import BaseClassifier
from .base.vectorizer import BaseVectorizer


class Surround(BaseVectorizer, BaseClassifier):

    def __init__(self, **kwargs):
        BaseVectorizer.__init__(self, vectorizer_name="spacy")
        BaseClassifier.__init__(self, **kwargs)

    def fit(  # type: ignore[override]
            self,
            samples: List[dict],
            y: np.ndarray,
            groups: List[str]
    ) -> None:
        sentences = self._get_surroundings(samples)
        self._vectorizer_fit(sentences)
        x = self._vectorizer_transform(sentences)
        BaseClassifier.fit(self, x, y)

    def predict(  # type: ignore[override]
            self,
            samples: List[dict],
            for_lime: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        sentences = self._get_surroundings(samples)
        x = self._vectorizer_transform(sentences)
        predictions_proba = BaseClassifier.predict(self, x)
        predictions = predictions_proba.argmax(axis=1)
        return predictions, predictions_proba

    @staticmethod
    def _get_surroundings(samples: List[dict]) -> List[str]:
        surroundings = []
        for sample in samples:
            surroundings.extend([
                " ".join(sample["tokens"][:sample["index_1"]]),
                " ".join(sample["tokens"][sample["index_1"]+1:sample["index_2"]]),
                " ".join(sample["tokens"][sample["index_2"]+1:])
            ])
        return surroundings

    def _vectorizer_fit(self, sentences: List[str]) -> None:
        BaseVectorizer.fit(self, sentences)

    def _vectorizer_transform(self, sentences: List[str]) -> np.ndarray:
        x_flatten = BaseVectorizer.transform(self, sentences)
        x = np.zeros((x_flatten.shape[0]//3, x_flatten.shape[1]*3))
        for i in range(x_flatten.shape[0]//3):
            x[i, :] = np.concatenate([
                x_flatten[(i*3), :],
                x_flatten[(i*3)+1, :],
                x_flatten[(i*3)+2, :]
            ])
        return x
