import numpy as np
from typing import List, Tuple
from .base.classifier import BaseClassifier
from .base.engineering import BaseEngineering


class Statistical(BaseEngineering, BaseClassifier):

    def __init__(self, **kwargs):
        BaseEngineering.__init__(self)
        BaseClassifier.__init__(self, **kwargs)

    def fit(  # type: ignore[override]
            self,
            samples: List[dict],
            y: np.ndarray,
            groups: List[str]
    ) -> None:
        BaseEngineering.fit(self, samples)
        x = BaseEngineering.transform(self, samples)
        BaseClassifier.fit(self, x, y)

    def predict(  # type: ignore[override]
            self,
            samples: List[dict],
            for_explainer: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = BaseEngineering.transform(self, samples)
        predictions_proba = BaseClassifier.predict(self, x)
        predictions = predictions_proba.argmax(axis=1)
        return predictions, predictions_proba
