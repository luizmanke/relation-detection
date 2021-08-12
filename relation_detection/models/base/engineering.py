import pandas as pd
import re
from typing import List


class BaseEngineering:

    def __init__(self) -> None:
        pass

    def get_features(self, samples: List[dict]) -> pd.DataFrame:
        features = []
        for sample in samples:

            new_features = {}
            for method_name in dir(self):
                if method_name[:9] == "_compute_":
                    method = getattr(self, method_name)
                    new_features[method_name[9:]] = method(sample)
            features.append(new_features)

        return pd.DataFrame(features)

    def _compute_length_before_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "before")
        return len(sentence)

    def _compute_length_between_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "between")
        return len(sentence)

    def _compute_length_after_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "after")
        return len(sentence)

    def _compute_n_tokens_before_entities(self, sample: dict) -> int:
        indexes = self._get_sorted_indexes(sample)
        return indexes[0]

    def _compute_n_tokens_between_entities(self, sample: dict) -> int:
        return abs(sample["index_1"] - sample["index_2"]) - 1

    def _compute_n_tokens_after_entities(self, sample: dict) -> int:
        indexes = self._get_sorted_indexes(sample)
        return len(sample["tokens"]) - indexes[1] - 1

    def _compute_entities_order(self, sample: dict) -> int:
        return 0 if sample["index_1"] > sample["index_2"] else 1

    def _compute_n_commas_before_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "before")
        return len(re.findall(",", sentence))

    def _compute_n_commas_between_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "between")
        return len(re.findall(",", sentence))

    def _compute_n_commas_after_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "after")
        return len(re.findall(",", sentence))

    def _compute_n_semicolons_before_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "before")
        return len(re.findall(";", sentence))

    def _compute_n_semicolons_between_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "between")
        return len(re.findall(";", sentence))

    def _compute_n_semicolons_after_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "after")
        return len(re.findall(";", sentence))

    def _compute_n_colons_before_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "before")
        return len(re.findall(":", sentence))

    def _compute_n_colons_between_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "between")
        return len(re.findall(":", sentence))

    def _compute_n_colons_after_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "after")
        return len(re.findall(":", sentence))

    def _compute_n_periods_before_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "before")
        return len(re.findall(r"\.", sentence))

    def _compute_n_periods_between_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "between")
        return len(re.findall(r"\.", sentence))

    def _compute_n_periods_after_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "after")
        return len(re.findall(r"\.", sentence))

    def _compute_n_exclamation_points_before_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "before")
        return len(re.findall("!", sentence))

    def _compute_n_exclamation_points_between_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "between")
        return len(re.findall("!", sentence))

    def _compute_n_exclamation_points_after_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "after")
        return len(re.findall("!", sentence))

    def _compute_n_question_marks_before_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "before")
        return len(re.findall(r"\?", sentence))

    def _compute_n_question_marks_between_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "between")
        return len(re.findall(r"\?", sentence))

    def _compute_n_question_marks_after_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "after")
        return len(re.findall(r"\?", sentence))

    def _compute_n_quotation_marks_before_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "before")
        return len(re.findall("'", sentence))

    def _compute_n_quotation_marks_between_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "between")
        return len(re.findall("'", sentence))

    def _compute_n_quotation_marks_after_entities(self, sample: dict) -> int:
        sentence = self._get_sentence(sample, "after")
        return len(re.findall("'", sentence))

    def _get_sentence(self, sample: dict, location: str) -> str:
        indexes = self._get_sorted_indexes(sample)
        if location == "before":
            sentence = " ".join(sample["tokens"][:indexes[0]])
        elif location == "between":
            sentence = " ".join(sample["tokens"][indexes[0]+1:indexes[1]])
        elif location == "after":
            sentence = " ".join(sample["tokens"][indexes[1]+1:])
        return sentence

    @staticmethod
    def _get_sorted_indexes(sample: dict) -> list:
        return sorted([sample["index_1"], sample["index_2"]])
