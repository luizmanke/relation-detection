from copy import deepcopy
from typing import Callable, List, Tuple

TOKEN_1 = "ENTITY1"  # nosec
TOKEN_2 = "ENTITY2"  # nosec


class BaseExplainer:

    def __init__(self):
        pass

    @staticmethod
    def _replace_entity_tokens(sample: dict) -> dict:
        sample = deepcopy(sample)
        sample["tokens"][sample["index_1"]] = TOKEN_1
        sample["tokens"][sample["index_2"]] = TOKEN_2
        return sample

    @staticmethod
    def _get_sentence(sample: dict) -> str:
        return " ".join(sample["tokens"])

    @staticmethod
    def _find_entities_index(tokens: List[str]) -> Tuple[int, int]:
        for i, token in enumerate(tokens):
            if token == TOKEN_1:
                index_1 = i
            if token == TOKEN_2:
                index_2 = i
        return index_1, index_2

    def _recreate_samples(
        self,
        sentences: List[str],
        index_1: int,
        index_2: int,
        tokenizer: Callable
    ) -> List[dict]:
        samples = []
        for sentence in sentences:
            tokens = tokenizer(sentence)
            tokens[index_1] = TOKEN_1
            tokens[index_2] = TOKEN_2
            samples.append({
                "tokens": tokens,
                "index_1": index_1,
                "index_2": index_2
            })
        return samples
