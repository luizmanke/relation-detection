from typing import Any
from .explainers.shap import SHAP


class Explainer:

    def __init__(self):
        pass

    def explain_sample(self, model: Any, sample: dict) -> None:
        SHAP().explain_sample(model, sample)
