import nltk
import numpy as np
import pandas as pd
from typing import Any, Dict


class News:

    def __init__(self, file_path: str) -> None:
        self.file_path_ = file_path
        self._download_nltk_model()
        self._load_text()
        self._extract_sentences()
        self._span_entities()
        self._tokenize()
        self._mark_entities()
        self._rename_relations()

    def get_data(self) -> Dict[str, Any]:
        KEYS = ["tokens", "index_1", "index_2", "relation", "content"]
        samples = [sample for sample in self.samples_ if sample["relation"] != -1]
        selected_samples = [
            {
                key: value for key, value in sample.items() if key in KEYS
            } for sample in samples
        ]
        selected_labels = np.array([sample.pop("relation") for sample in selected_samples])
        selected_groups = [sample.pop("content") for sample in selected_samples]
        return {
            "samples": selected_samples,
            "labels": selected_labels,
            "groups": selected_groups
        }

    @staticmethod
    def _download_nltk_model() -> None:
        try:
            nltk.data.find("corpora/stopwords")
            nltk.data.find("tokenizers/punkt")
        except Exception:
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)

    def _load_text(self) -> None:
        self.text_ = pd.read_csv(self.file_path_)

    def _extract_sentences(self) -> None:
        self.samples_ = self.text_.to_dict(orient="records")

    def _span_entities(self) -> None:
        for sample in self.samples_:
            spans = sorted([
                [sample["start_index_nm"], sample["end_index_nm"]],
                [sample["start_index_kw"], sample["end_index_kw"]]
            ])
            sample["spans"] = [
                sample["content"][:spans[0][0]],
                sample["content"][spans[0][0]:spans[0][1]],
                sample["content"][spans[0][1]:spans[1][0]],
                sample["content"][spans[1][0]:spans[1][1]],
                sample["content"][spans[1][1]:]
            ]

    def _tokenize(self) -> None:
        for sample in self.samples_:
            sample["tokens"] = []
            for span in sample["spans"]:
                if span == "":
                    continue
                if span not in [sample["name"], sample["keyword"]]:
                    sample["tokens"].extend(nltk.word_tokenize(span, language="portuguese"))
                else:
                    sample["tokens"].append(span)

    def _mark_entities(self) -> None:
        for sample in self.samples_:
            for index, token in enumerate(sample["tokens"]):
                if token == sample["name"] and "index_1" not in sample:
                    sample["index_1"] = index
                elif token == sample["keyword"]:
                    sample["index_2"] = index
                if "index_1" in sample and "index_2" in sample:
                    break

    def _rename_relations(self) -> None:
        for sample in self.samples_:
            sample["relation"] = sample["label"]
