from typing import Any, List
from .explainers.lime import LIME


class Explainer:

    def __init__(self):
        pass

    def explain_samples(self, model: Any, samples: List[dict]) -> None:
        for sample in samples:
            LIME().explain_sample(model, sample)
