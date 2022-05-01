import numpy as np
from typing import List, Tuple
from .base.classifier import Classifier
from .base.embedder import Embedder


class Surround(Embedder, Classifier):

    def __init__(self, **kwargs):
        Embedder.__init__(self)
        Classifier.__init__(self, **kwargs)

    def fit(  # type: ignore[override]
            self,
            samples: List[dict],
            y: np.ndarray
    ) -> None:
        sentences = self._get_surroundings(samples)
        x = self._vectorizer_transform(sentences)
        Classifier.fit(self, x, y)

    def predict(  # type: ignore[override]
            self,
            samples: List[dict],
            for_explainer: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        sentences = self._get_surroundings(samples)
        x = self._vectorizer_transform(sentences)
        predictions_proba = Classifier.predict(self, x)
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

    def _vectorizer_transform(self, sentences: List[str]) -> np.ndarray:
        x_flatten = Embedder.transform(self, sentences)
        x = np.zeros((x_flatten.shape[0]//3, x_flatten.shape[1]*3))
        for i in range(x_flatten.shape[0]//3):
            x[i, :] = np.concatenate([
                x_flatten[(i*3), :],
                x_flatten[(i*3)+1, :],
                x_flatten[(i*3)+2, :]
            ])
        return x
