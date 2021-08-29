import numpy as np
import shap
from typing import Any, List
from .base.explainer import BaseExplainer


class SHAP(BaseExplainer):

    def __init__(self):
        pass

    def explain_samples(self, model: Any, samples: List[dict]) -> None:
        if len(samples) == 1:
            shap_values = self._explain_sample(model, samples[0])
        else:
            shap_values = self._loop_samples(model, samples)
        for i in range(len(shap_values)):
            shap.plots.text(shap_values[i, :, 1])

    def explain_model(self, model: Any, samples: List[dict]) -> None:
        shap_values = self._loop_samples(model, samples)
        shap.plots.bar(shap_values[:, :, 1], max_display=20)

    def _loop_samples(self, model: Any, samples: List[dict]) -> shap.Explanation:
        shap_values = [self._explain_sample(model, sample) for sample in samples]
        return shap.Explanation(
            np.array([x.values[0, :, :] for x in shap_values], dtype=object),
            np.array([x.base_values[0, :] for x in shap_values], dtype=object),
            np.array([x.data[0, :] for x in shap_values], dtype=object),
            feature_names=[x.feature_names[0] for x in shap_values],
            clustering=np.array([x.clustering[0, :, :] for x in shap_values])
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
            return model.predict(samples, return_proba=True, for_explainer=True)

        sentence_without_entities = self._create_sentence_without_entities(sample)
        explainer = shap.Explainer(_nested_predict, self.tokenizer_)
        shap_value = explainer([sentence_without_entities], silent=True)
        self._replace_values(shap_value, sample, index_1, index_2)
        return shap.Explanation(
            shap_value.values,
            shap_value.base_values,
            shap_value.data,
            feature_names=shap_value.feature_names,
            clustering=shap_value.clustering
        )

    @staticmethod
    def _replace_values(
            shap_value: shap.Explanation,
            sample: dict,
            index_1: int,
            index_2: int
    ) -> None:
        entity_1 = "[E1] " + sample["tokens"][sample["index_1"]] + " [/E1] "
        entity_2 = "[E2] " + sample["tokens"][sample["index_2"]] + " [/E2] "
        index_1 = index_1 - 1 if index_1 > index_2 else index_1
        shap_value.values = np.insert(shap_value.values, index_1+1, [0, 0], axis=1)
        shap_value.values = np.insert(shap_value.values, index_2+1, [0, 0], axis=1)
        shap_value.data = shap_value.data.astype(object)
        shap_value.data = np.insert(shap_value.data, index_1+1, entity_1, axis=1)
        shap_value.data = np.insert(shap_value.data, index_2+1, entity_2, axis=1)
