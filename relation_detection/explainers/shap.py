import numpy as np
import shap
from typing import Any, List
from .base.explainer import BaseExplainer


class SHAP(BaseExplainer):

    def __init__(self):
        pass

    def explain_samples(self, model: Any, samples: List[dict]) -> None:
        shap_values = self._loop_samples(model, samples)
        shap.plots.text(shap_values[:, :, 1])

    def explain_model(self, model: Any, samples: List[dict]) -> None:
        shap_values = self._loop_samples(model, samples)
        shap.plots.bar(shap_values[:, :, 1])

    def _loop_samples(self, model: Any, samples: List[dict]) -> shap.Explanation:
        values: List[str] = []
        base_values: List[str] = []
        data: List[str] = []
        feature_names: List[str] = []
        hierarchical_values: List[str] = []
        clustering: List[str] = []

        for sample in samples:
            raw_shap_value = self._explain_sample(model, sample)
            values.append(raw_shap_value.values[0, :, :])
            base_values.append(raw_shap_value.base_values[0, :])
            data.append(raw_shap_value.data[0, :])
            feature_names.append(raw_shap_value.feature_names[0])
            hierarchical_values.append(raw_shap_value.hierarchical_values[0, :, :])
            clustering.append(raw_shap_value.clustering[0, :, :])

        return shap.Explanation(
            np.array(values, dtype=object),
            np.array(base_values, dtype=object),
            np.array(data, dtype=object),
            feature_names=feature_names,
            hierarchical_values=np.array(hierarchical_values),
            clustering=np.array(clustering)
        )

    def _set_tokenizer(self, model: Any) -> None:
        self.tokenizer_ = model.model_.tokenizer_
        self._tokenize_to_words = self.tokenizer_.tokenize

    def _explain_sample(self, model: Any, sample: dict) -> shap.Explanation:
        self._set_tokenizer(model)
        original_tokens = self._get_original_tokens(sample)
        index_1, index_2 = self._find_entities_index(original_tokens)

        # define nested function
        def _nested_predict(sentences: List[str]) -> np.ndarray:
            samples = self._recreate_samples(sentences, index_1, index_2)
            return model.predict(samples, return_proba=True, for_lime=True)

        sentence_without_entities = self._create_sentence_without_entities(sample)
        explainer = shap.Explainer(_nested_predict, self.tokenizer_)
        return explainer([sentence_without_entities], silent=True)
