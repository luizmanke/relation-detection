from typing import Any, List
from .explainers.lime import LIME
# from .explainers.shap import SHAP


class Explainer:

    available_explainers_ = ["lime", "shap"]

    def __init__(self):
        pass

    def explain_sample(self, model: Any, sample: dict, kind: str) -> None:
        assert kind in self.available_explainers_
        if kind == "lime":
            LIME().explain_sample(model, sample)
        if kind == "shap":
            pass

    def explain_model(self, model: Any, sample: List[dict], kind: str) -> None:
        assert kind in self.available_explainers_
        if kind == "lime":
            LIME().explain_model(model, sample)
        if kind == "shap":
            pass
