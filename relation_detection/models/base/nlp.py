import random
import spacy
from typing import List, Tuple


class NLP:

    def __init__(self) -> None:
        self.nlp_ = spacy.load("pt_core_news_lg")

    def extract_info(self, samples: List[dict], word_dropout: bool = False) -> List[dict]:
        info: List[dict] = []

        from tqdm import tqdm
        for sample in tqdm(samples):
            doc = self._create_doc(sample)
            nlp = self._get_nlp(doc)
            indexes = self._get_entities_indexes(sample, doc)
            tokens_masked = self._mask_entities(nlp["token"], indexes)
            positions = self._get_positions(len(tokens_masked), indexes)
            tokens_dropout = self._word_dropout(tokens_masked, word_dropout)
            info.append({
                "tokens": tokens_dropout,
                "heads": nlp["head"],
                "dependencies": nlp["dep"],
                "part_of_speeches": nlp["pos"],
                "types": nlp["ner"],
                "position_1": positions["entity_1"],
                "position_2": positions["entity_2"]
            })
        return info

    def _create_doc(self, sample: dict):
        return self.nlp_(" ".join(sample["tokens"]))

    def _get_nlp(self, doc) -> dict:
        word, head, dep, pos, ner = [], [], [], [], []
        for token in doc:
            word.append(self.nlp_.vocab.strings[token.text])
            head.append(token.head.i)
            dep.append(token.dep)
            pos.append(token.pos)
            ner.append(token.ent_type)
        return {"token": word, "head": head, "dep": dep, "pos": pos, "ner": ner}

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
        vocab = self.nlp_.vocab.strings
        tokens = [token for token in tokens]
        index_start_1 = indexes["index_start_1"]
        index_end_1 = indexes["index_end_1"]
        index_start_2 = indexes["index_start_2"]
        index_end_2 = indexes["index_end_2"]
        tokens[index_start_1:index_end_1+1] = [vocab["E1"]] * (index_end_1 - index_start_1 + 1)
        tokens[index_start_2:index_end_2+1] = [vocab["E2"]] * (index_end_2 - index_start_2 + 1)
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
                word if random.random() > DROPOUT_CHANCE else self.nlp_.vocab.strings["UNKNOWN"]
                for word in words
            ]
        return words_dropout
