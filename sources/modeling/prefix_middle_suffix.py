import numpy as np
import os
import pickle
from typing import List
from .base.classifier import BaseClassifier
from .base.vectorizer import BaseVectorizer


class PrefixMiddleSuffix(BaseVectorizer, BaseClassifier):

    def __init__(self, **kwargs):
        BaseVectorizer.__init__(self, vectorizer_name="spacy")
        BaseClassifier.__init__(self, **kwargs)

    def fit(self, samples: List[dict], y: np.ndarray) -> None:
        sentences = self._get_prefix_middle_suffix(samples)
        self._vectorizer_fit(sentences)
        x = self._vectorizer_transform(sentences)
        BaseClassifier.fit(self, x, y)

    def predict(self, samples: List[dict]) -> np.ndarray:
        sentences = self._get_prefix_middle_suffix(samples)
        x = self._vectorizer_transform(sentences)
        return BaseClassifier.predict(self, x)

    def save(self, dir: str) -> None:
        if not os.path.isdir(dir):
            os.makedirs(dir)
        with open(f"{dir}/model.pickle", "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dir: str) -> None:
        with open(f"{dir}/model.pickle", "rb") as file:
            self.__dict__.update(pickle.load(file))

    @staticmethod
    def _get_prefix_middle_suffix(samples: List[dict]) -> List[str]:
        prefix_middle_suffix = []
        for sample in samples:
            prefix_middle_suffix.extend([
                " ".join(sample["tokens"][:sample["index_1"]]),
                " ".join(sample["tokens"][sample["index_1"]+1:sample["index_2"]]),
                " ".join(sample["tokens"][sample["index_2"]+1:])
            ])
        return prefix_middle_suffix

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
