from typing import Any
from .explainers.lime import LIME


class Explainer:

    def __init__(self):
        pass

    def explain_sample(self, model: Any, sample: dict) -> None:
        LIME().explain_sample(model, sample)
