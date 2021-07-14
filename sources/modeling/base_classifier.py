import numpy as np
from sklearn.ensemble import RandomForestClassifier


class BaseClassifier(RandomForestClassifier):

    def __init__(self, **kwargs) -> None:
        RandomForestClassifier.__init__(self, **kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        RandomForestClassifier.fit(self, x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return RandomForestClassifier.predict(self, x)
