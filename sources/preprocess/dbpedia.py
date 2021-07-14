import numpy as np
import re
from imblearn.under_sampling import RandomUnderSampler
from typing import Any, Dict, List, Tuple


class DBpedia:

    def __init__(self, file_name: str):
        self.file_name = file_name

    def get_data(self) -> List[dict]:
        self._load_text()
        self._extract_sentences()
        self._span_entities()
        self._tokenize()
        self._mark_entities()
        self._rename_relations()
        return self._get_data()

    def _load_text(self) -> None:
        with open(self.file_name, "r") as text_file:
            self.text = text_file.read()

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

        self.samples = []
        sample: Dict[str, Any] = {}
        for line in self.text.split("\n"):

            if line.find(":") > 0:  # content
                key = KEYS[len(sample)]
                value = line.split(":")[-1][1:]
                sample[key] = value

            if line.find("****") == 0:  # break
                if sample:
                    self.samples.append(sample)
                    sample = {}

    def _span_entities(self) -> None:
        for sample in self.samples:
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
        for sample in self.samples:
            if "spans" not in sample:
                continue

            sample["tokens"] = []
            for span in sample["spans"]:
                if span == "":
                    continue
                if span not in [sample["ENTITY1"], sample["ENTITY2"]]:
                    sample["tokens"].extend(span.split())
                else:
                    sample["tokens"].append(span)

    def _mark_entities(self) -> None:
        for sample in self.samples:
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
        for sample in self.samples:
            sample["relation"] = 0 if sample["REL TYPE"] == "other" else 1

    def _get_data(self) -> Tuple[List[dict], np.ndarray]:
        samples = [sample for sample in self.samples if "tokens" in sample]

        # resample
        labels = np.array([sample["relation"] for sample in samples]).reshape(-1, 1)
        indexes = np.array([i for i, _ in enumerate(labels)]).reshape(-1, 1)
        new_indexes, _ = RandomUnderSampler(random_state=42).fit_resample(indexes, labels)
        new_indexes = sorted(new_indexes[:, 0])

        # select samples
        KEYS = ["tokens", "index_1", "index_2", "relation"]
        selected_samples = [
            {
                key: value for key, value in samples[i].items() if key in KEYS
            } for i in new_indexes
        ]
        selected_y = np.array([sample.pop("relation") for sample in selected_samples])

        return selected_samples, selected_y
