from copy import deepcopy
from typing import Callable, List, Tuple


class BaseExplainer:

    def __init__(self):
        self._tokenize_to_words = Callable

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
