from typing import Any, List
from .explainers.lime import LIME
from .explainers.shap import SHAP


class Explainer:

    available_explainers_ = ["shap", "lime"]

    def __init__(self):
        pass

    def explain_samples(self, model: Any, samples: List[dict], kind: str = "shap") -> None:
        assert kind in self.available_explainers_
        if kind == "shap":
            SHAP().explain_samples(model, samples)
        if kind == "lime":
            for sample in samples:
                LIME().explain_sample(model, sample)

    def explain_model(self, model: Any, samples: List[dict], kind: str = "shap") -> None:
        assert kind in self.available_explainers_
        if kind == "shap":
            SHAP().explain_model(model, samples)
        if kind == "lime":
            LIME().explain_model(model, samples)
