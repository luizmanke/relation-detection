import nltk
import numpy as np
from lime.submodular_pick import SubmodularPick
from lime.lime_text import LimeTextExplainer
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

        sentence = self._create_sentence_without_entities(sample)
        self._explain_sample(sentence, _nested_predict)

    def explain_model(self, model: Any, samples: List[dict]) -> None:
        globals()["i"] = -1
        lookup_table = [
            self._find_entities_index(
                self._get_original_tokens(sample)
            ) for sample in samples
        ]

        # define nested function
        def _nested_predict(sentences: List[str]) -> np.ndarray:
            globals()["i"] += 1
            index_1, index_2 = lookup_table[globals()["i"]]
            samples = self._recreate_samples(sentences, index_1, index_2)
            return model.predict(samples, return_proba=True, for_lime=True)

        sentences = [self._create_sentence_without_entities(sample) for sample in samples]
        self._explain_model(sentences, _nested_predict)

    def _get_original_tokens(self, sample: dict) -> List[str]:
        sample = dict(sample)
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
        sample = dict(sample)
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

    def _explain_sample(self, sentence: str, predict_function: Callable) -> None:
        explainer = self._create_explainer()
        lime_values = explainer.explain_instance(
            text_instance=sentence,
            classifier_fn=predict_function
        )
        lime_values.show_in_notebook(
            text=True,
            labels=(lime_values.available_labels()[0],)
        )

    def _explain_model(self, sentences: List[str], predict_function: Callable) -> None:
        explainer = self._create_explainer()
        picks = SubmodularPick(
            explainer,
            sentences,
            predict_function,
            method="full",
            num_exps_desired=2
        )
        for lime_values in picks.sp_explanations:
            lime_values.show_in_notebook(
                text=True,
                labels=(lime_values.available_labels()[0],)
            )

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
