import pandas as pd
import re


class DBpedia:

    def __init__(self):
        self._clear_attributes()

    def get_samples(self, file_name):
        self._clear_attributes(file_name=file_name)
        self._load_text()
        self._extract_sentences()
        self._span_entities()
        self._tokenize()
        self._mark_entities()
        return self._get_dataframe()

    def _clear_attributes(self, **kwargs):
        self.file_name = ""
        self.text = ""
        self.samples = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _load_text(self):
        with open(self.file_name, "r") as text_file:
            self.text = text_file.read()

    def _extract_sentences(self):
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
        sample = {}
        for line in self.text.split("\n"):

            if line.find(":") > 0:  # content
                key = KEYS[len(sample)]
                value = line.split(":")[-1][1:]
                sample[key] = value

            if line.find("****") == 0:  # break
                if sample:
                    self.samples.append(sample)
                    sample = {}

    def _span_entities(self):
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

                span_1 = re.search(entity_1, sample["SENTENCE"]).span()
                span_2 = re.search(entity_2, sample["SENTENCE"]).span()
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

    def _tokenize(self):
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

    def _mark_entities(self):
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

    def _get_dataframe(self):
        COLUMNS = ["tokens", "index_1", "index_2"]
        samples = [sample for sample in self.samples if "tokens" in sample]
        df = pd.DataFrame(samples).astype({"index_1": int, "index_2": int})
        return df[COLUMNS]
