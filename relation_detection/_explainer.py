from typing import Any, List

from .explainers.shap import SHAP


class Explainer:

    def __init__(self):
        pass

    def explain_sample(self, model: Any, sample: dict) -> None:
        SHAP().explain_sample(model, sample)

    def explain_model(self, model: Any, samples: List[dict]) -> None:
        SHAP().explain_model(model, samples)
