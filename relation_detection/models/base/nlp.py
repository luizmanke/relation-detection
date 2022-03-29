import numpy as np
import random
import spacy
from typing import Dict, List, Tuple


class NLP:

    PAD_TOKEN = "[PAD]"
    UNKNOWN_TOKEN = "[UNKNOWN]"
    E1_TOKEN = "[E1]"
    E2_TOKEN = "[E2]"

    def __init__(self) -> None:
        self.nlp_ = spacy.load("pt_core_news_lg")
        self.map_: Dict[str, dict] = {}
        self.vectors_ = np.empty(0)

    def fit(self, samples: List[dict]) -> None:

        # get values
        list_of_dicts = []
        for sample in samples:
            doc = self._create_doc(sample)
            list_of_dicts.append(self._get_nlp(doc))

        # create map
        base_values = [self.PAD_TOKEN, self.UNKNOWN_TOKEN, self.E1_TOKEN, self.E2_TOKEN]
        for key in ["words", "dependencies", "part_of_speeches", "types"]:
            values = []
            for item in list_of_dicts:
                values.extend(item[key])
            all_values = base_values + sorted(set(values))
            self.map_[key] = {value: i for i, value in enumerate(all_values)}

        # create vector
        vectors_length = len(self.map_["words"])
        vectors_size = self.nlp_.vocab.vectors.shape[1]
        self.vectors_ = np.random.uniform(-1, 1, (vectors_length, vectors_size))
        for word, i in self.map_["words"].items():
            self.vectors_[i, :] = self.nlp_.vocab[word].vector
        self.vectors_[self.map_["words"][self.PAD_TOKEN]] = 0  # ensure pad vector is zero

    def extract(self, samples: List[dict], word_dropout: bool = False) -> List[dict]:
        info: List[dict] = []
        for sample in samples:
            doc = self._create_doc(sample)
            nlp = self._get_nlp(doc)
            nlp_ids = self._text_to_id(nlp)
            indexes = self._get_entities_indexes(sample, doc)
            tokens_masked = self._mask_entities(nlp_ids["words"], indexes)
            positions = self._get_positions(len(tokens_masked), indexes)
            tokens_dropout = self._word_dropout(tokens_masked, word_dropout)
            info.append({
                "words": tokens_dropout,
                "heads": nlp_ids["heads"],
                "dependencies": nlp_ids["dependencies"],
                "part_of_speeches": nlp_ids["part_of_speeches"],
                "types": nlp_ids["types"],
                "position_1": positions["entity_1"],
                "position_2": positions["entity_2"]
            })
        return info

    def _create_doc(self, sample: dict):
        return self.nlp_(" ".join(sample["tokens"]))

    def _get_nlp(self, doc) -> dict:
        return {
            "words": [token.text for token in doc],
            "heads": [token.head.i for token in doc],
            "dependencies": [token.dep for token in doc],
            "part_of_speeches": [token.pos for token in doc],
            "types": [token.ent_type for token in doc]
        }

    def _text_to_id(self, nlp: dict) -> dict:
        assert self.map_ != {}
        nlp = {**nlp}
        for key in self.map_:
            map = self.map_[key]
            nlp[key] = [map[x] if x in map else map[self.UNKNOWN_TOKEN] for x in nlp[key]]
        return nlp

    def _get_entities_indexes(self, sample: dict, doc) -> dict:
        name_1 = sample["tokens"][sample["index_1"]]
        name_2 = sample["tokens"][sample["index_2"]]
        index_start_1, index_end_1 = self._find_name_indexes(name_1, doc)
        index_start_2, index_end_2 = self._find_name_indexes(name_2, doc)
        return {
            "index_start_1": index_start_1,
            "index_end_1": index_end_1,
            "index_start_2": index_start_2,
            "index_end_2": index_end_2
        }

    @staticmethod
    def _find_name_indexes(name: str, doc) -> Tuple[int, int]:

        name_stripped = name.strip()
        text = doc.text
        start_char = text.find(name_stripped)
        end_char = start_char + len(name_stripped)

        for token in doc:
            if token.idx == start_char:
                index_start = token.i
            if token.idx + len(token) == end_char:
                index_end = token.i

        return index_start, index_end

    def _mask_entities(self, tokens: List[int], indexes: dict) -> List[int]:
        tokens = [token for token in tokens]
        index_start_1 = indexes["index_start_1"]
        index_end_1 = indexes["index_end_1"]
        index_start_2 = indexes["index_start_2"]
        index_end_2 = indexes["index_end_2"]
        tokens[index_start_1:index_end_1+1] = [self.map_["words"]["[E1]"]] * (index_end_1 - index_start_1 + 1)
        tokens[index_start_2:index_end_2+1] = [self.map_["words"]["[E2]"]] * (index_end_2 - index_start_2 + 1)
        return tokens

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.nlp_.vocab.strings[x] for x in tokens]

    @staticmethod
    def _get_positions(length: int, indexes: dict) -> dict:

        entities = [
            (indexes["index_start_1"], indexes["index_end_1"]),
            (indexes["index_start_2"], indexes["index_end_2"])
        ]

        positions = []
        for entity in entities:
            before = list(range(-entity[0], 0))
            middle = [0] * (entity[1] - entity[0] + 1)
            after = list(range(1, length - entity[1]))
            positions.append(before + middle + after)

        return {
            "entity_1": positions[0],
            "entity_2": positions[1]
        }

    def _word_dropout(self, words: List[int], drop: bool) -> List[int]:
        DROPOUT_CHANCE = 0.04
        words_dropout = [word for word in words]
        if drop:
            random.seed(42)
            words_dropout = [
                word if random.random() > DROPOUT_CHANCE else self.map_["words"][self.UNKNOWN_TOKEN]
                for word in words
            ]
        return words_dropout
