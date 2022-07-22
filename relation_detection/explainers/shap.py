from typing import Any, List

import numpy as np
import shap
import spacy

from .base.explainer import BaseExplainer


class SHAP(BaseExplainer):

    def __init__(self):
        self.nlp_ = spacy.load("pt_core_news_lg")

    def explain_sample(self, model: Any, sample: dict) -> None:
        assert isinstance(sample, dict)
        shap_values = self._explain_sample(model, sample)
        shap.plots.text(shap_values[0, :, 1])

    def _explain_sample(self, model: Any, sample: dict) -> shap.Explanation:
        sentence = " ".join(sample["tokens"])
        sample_replaced = self._replace_entity_tokens(sample)
        sentence_replaced = self._get_sentence(sample_replaced)
        tokens_replaced = self._tokenize(sentence_replaced)["input_ids"]
        index_1, index_2 = self._find_entities_index(tokens_replaced)

        # define nested function
        def _nested_predict(sentences: List[str]) -> np.ndarray:
            sentences = [str(s) for s in sentences]
            samples = self._recreate_samples(sentences, index_1, index_2, lambda x: self._tokenize(x)["input_ids"])
            return model.predict(samples, return_proba=True, for_explainer=True)

        # mask token with trailing spaces to prevent words concatenation during join
        masker = shap.maskers.Text(self._tokenize, mask_token=" MASK_TOKEN ", collapse_mask_token=False)
        explainer = shap.Explainer(_nested_predict, masker)
        shap_values = explainer([sentence], silent=True)
        self._replace_values(shap_values, sample, index_1, index_2)
        return shap_values

    @staticmethod
    def _replace_values(
        shap_value: shap.Explanation,
        sample: dict,
        index_1: int,
        index_2: int
    ) -> None:
        shap_value.data[0][index_1] = f" [E1] {shap_value.data[0][index_1]} [/E1] "
        shap_value.data[0][index_2] = f" [E2] {shap_value.data[0][index_2]} [/E2] "

    def _tokenize(self, s: str, return_offsets_mapping=True) -> dict:

        doc = self.nlp_(s)
        input_ids, offset_mapping = [], []
        for token in doc:
            input_ids.append(token.text)
            offset_mapping.append((token.idx, token.idx+len(token.text)))

        output = {"input_ids": input_ids}
        if return_offsets_mapping:
            output["offset_mapping"] = offset_mapping

        return output
