import nltk
import numpy as np
import re
from imblearn.under_sampling import RandomUnderSampler
from typing import Any, Dict, List, Tuple
from .._dataset import BaseDataset


class DBpedia(BaseDataset):

    def __init__(self, file_path: str):
        self.file_path_ = file_path
        self._load_text()
        self._extract_sentences()
        self._span_entities()
        self._tokenize()
        self._mark_entities()
        self._rename_relations()

    def get_data(self) -> Tuple[List[dict], np.ndarray, List[str]]:
        filtered_samples = self._filter_samples()
        selected_samples = self._resample(filtered_samples)
        return self._get_data(selected_samples)

    def _load_text(self) -> None:
        with open(self.file_path_, "r", encoding="utf-8") as text_file:
            self.text_ = text_file.read()

    def _extract_sentences(self) -> None:
        KEYS = [
            "SENTENCE",
            "MANUALLY CHECKED",
            "ENTITY1",
            "TYPE1",
            "ENTITY2",
            "TYPE2",
            "REL TYPE"
        ]

        self.samples_ = []
        sample: Dict[str, Any] = {}
        for line in self.text_.split("\n"):

            if line.find(":") > 0:  # content
                key = KEYS[len(sample)]
                value = line.split(":")[-1][1:]
                sample[key] = value

            if line.find("****") == 0:  # break
                if sample:
                    self.samples_.append(sample)
                    sample = {}

    def _span_entities(self) -> None:
        for sample in self.samples_:
            entity_1 = sample["ENTITY1"].replace(".", r"\.")
            entity_2 = sample["ENTITY2"].replace(".", r"\.")

            # ignore inconsistent samples
            n_entity_1 = len(re.findall(entity_1, sample["SENTENCE"]))
            n_entity_2 = len(re.findall(entity_2, sample["SENTENCE"]))
            if (
                    (n_entity_1 == 1 and n_entity_2 == 1) and
                    (entity_1 != entity_2)
            ):

                search_1 = re.search(entity_1, sample["SENTENCE"])
                search_2 = re.search(entity_2, sample["SENTENCE"])
                if search_1 is not None and search_2 is not None:
                    span_1 = search_1.span()
                    span_2 = search_2.span()
                    if (
                            not (span_2[0] < span_1[0] < span_2[1]) and
                            not (span_2[0] < span_1[1] < span_2[1]) and
                            not (span_1[0] < span_2[0] < span_1[1]) and
                            not (span_1[0] < span_2[1] < span_1[1])
                    ):
                        spans = sorted(list(span_1) + list(span_2))
                        sample["spans"] = [
                            sample["SENTENCE"][:spans[0]],
                            sample["SENTENCE"][spans[0]:spans[1]],
                            sample["SENTENCE"][spans[1]:spans[2]],
                            sample["SENTENCE"][spans[2]:spans[3]],
                            sample["SENTENCE"][spans[3]:]
                        ]

    def _tokenize(self) -> None:
        for sample in self.samples_:
            if "spans" not in sample:
                continue

            sample["tokens"] = []
            for span in sample["spans"]:
                if span == "":
                    continue
                if span not in [sample["ENTITY1"], sample["ENTITY2"]]:
                    sample["tokens"].extend(nltk.word_tokenize(span, language="portuguese"))
                else:
                    sample["tokens"].append(span)

    def _mark_entities(self) -> None:
        for sample in self.samples_:
            if "tokens" not in sample:
                continue

            for index, token in enumerate(sample["tokens"]):
                if token == sample["ENTITY1"]:
                    sample["index_1"] = index
                elif token == sample["ENTITY2"]:
                    sample["index_2"] = index
                if "index_1" in sample and "index_2" in sample:
                    break

    def _rename_relations(self) -> None:
        for sample in self.samples_:
            sample["relation"] = 0 if sample["REL TYPE"] == "other" else 1

    def _filter_samples(self) -> List[dict]:
        return [sample for sample in self.samples_ if "tokens" in sample]

    @staticmethod
    def _resample(samples: List[dict]) -> List[dict]:
        labels = np.array([sample["relation"] for sample in samples]).reshape(-1, 1)
        indexes = np.array([i for i, _ in enumerate(labels)]).reshape(-1, 1)
        selected_indexes, _ = RandomUnderSampler(random_state=42).fit_resample(indexes, labels)
        selected_indexes = sorted(selected_indexes[:, 0])
        return [samples[i] for i in selected_indexes]

    def _get_data(self, samples: List[dict]) -> Tuple[List[dict], np.ndarray, List[str]]:
        KEYS = ["tokens", "index_1", "index_2", "relation", "SENTENCE"]
        selected_samples = [
            {
                key: value for key, value in sample.items() if key in KEYS
            } for sample in samples
        ]
        selected_labels = np.array([sample.pop("relation") for sample in selected_samples])
        selected_groups = [sample.pop("SENTENCE") for sample in selected_samples]
        return selected_samples, selected_labels, selected_groups
