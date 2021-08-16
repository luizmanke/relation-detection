import nltk
import numpy as np
from copy import deepcopy
from lime.explanation import Explanation
from lime.lime_text import IndexedString, LimeTextExplainer, TextDomainMapper
from typing import Any, Callable, List, Tuple
from . import utils


class Explainer:

    def __init__(self):
        utils.download_nltk_model()

    def explain_sample(self, model: Any, sample: dict) -> None:
        original_tokens = self._get_original_tokens(sample)
        index_1, index_2 = self._find_entities_index(original_tokens)

        # define nested function
        def _nested_predict(sentences: List[str]) -> np.ndarray:
            samples = self._recreate_samples(sentences, index_1, index_2)
            return model.predict(samples, return_proba=True, for_lime=True)

        sentence_without_entities = self._create_sentence_without_entities(sample)
        self._explain_sample(
            sentence_without_entities,
            _nested_predict,
            sample
        )

    def _get_original_tokens(self, sample: dict) -> List[str]:
        sample = deepcopy(sample)
        sample["tokens"][sample["index_1"]] = "[E1]"
        sample["tokens"][sample["index_2"]] = "[E2]"
        original_sentence = " ".join(sample["tokens"])
        return self._tokenize_to_words(original_sentence)

    @staticmethod
    def _find_entities_index(tokens: List[str]) -> Tuple[int, int]:
        for i, token in enumerate(tokens):
            if token == "[E1]":
                index_1 = i
            if token == "[E2]":
                index_2 = i
        return index_1, index_2

    @staticmethod
    def _create_sentence_without_entities(sample: dict) -> str:
        sample = deepcopy(sample)
        sample["tokens"][sample["index_1"]] = ""
        sample["tokens"][sample["index_2"]] = ""
        return " ".join(sample["tokens"])

    def _recreate_samples(self, sentences: List[str], index_1: int, index_2: int) -> List[dict]:
        samples = []
        for sentence in sentences:
            tokens = self._tokenize_to_words(sentence)
            for index, element in sorted(zip([index_1, index_2], ["[E1]", "[E2]"])):
                tokens.insert(index, element)
            samples.append({
                "tokens": tokens,
                "index_1": index_1,
                "index_2": index_2
            })
        return samples

    def _explain_sample(
            self,
            sentence: str,
            predict_function: Callable,
            sample: dict
    ) -> None:
        explainer = self._create_explainer()
        lime_values = explainer.explain_instance(
            text_instance=sentence,
            classifier_fn=predict_function
        )
        lime_values = self._replace_values(lime_values, sample)
        lime_values.show_in_notebook(
            text=True,
            labels=(lime_values.available_labels()[0],)
        )

    def _replace_values(self, lime_values: Explanation, sample: dict) -> Explanation:
        sample = deepcopy(sample)

        # get values
        tokens = sample["tokens"]
        index_1 = sample["index_1"]
        index_2 = sample["index_2"]

        # set entities
        entity_1 = "[E1] " + tokens[index_1] + " [E1]"
        entity_2 = "[E2] " + tokens[index_2] + " [E2]"

        # compute lengths
        length_1 = len(self._tokenize_to_words(entity_1))
        length_2 = len(self._tokenize_to_words(entity_2))

        # recompute
        tokens[index_1] = entity_1
        tokens[index_2] = entity_2
        new_index_2 = index_2 + length_1 - 1

        # replace values
        lime_values.domain_mapper = TextDomainMapper(IndexedString(
            " ".join(tokens),
            split_expression=self._tokenize_to_words,
            mask_string="[PAD]",
            bow=False
        ))
        new_local_exp = {1: [[x[0], x[1]] for x in lime_values.local_exp[1]]}
        for item in new_local_exp[1]:
            for index, length in sorted(zip([index_1, new_index_2], [length_1, length_2])):
                if item[0] >= index:
                    item[0] += length
        lime_values.local_exp = new_local_exp

        return lime_values

    def _create_explainer(self) -> LimeTextExplainer:
        return LimeTextExplainer(
            class_names=["not related", "related"],
            split_expression=self._tokenize_to_words,
            mask_string="[PAD]",
            bow=False
        )

    def _tokenize_to_words(self, sentence: str) -> List[str]:
        KNOWN_TOKENS = ["E1", "E2", "PAD"]

        i = 0
        tokens = []
        raw_tokens = nltk.word_tokenize(sentence, language="portuguese")
        while i < len(raw_tokens):
            if raw_tokens[i] == "[" and raw_tokens[i+1] in KNOWN_TOKENS and raw_tokens[i+2] == "]":
                tokens.append(f"[{raw_tokens[i+1]}]")
                i += 3
            else:
                tokens.append(raw_tokens[i])
                i += 1

        return tokens
