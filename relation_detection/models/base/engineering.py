import joblib
import math
import nltk
import numpy as np
import os
import pandas as pd
import re
import spacy
from collections import Counter
from sklearn.decomposition import PCA
from typing import List
from ...utils import download_nltk_model, download_spacy_model


class BaseEngineering:

    def __init__(self) -> None:
        download_nltk_model()
        download_spacy_model()
        self.stopwords_ = nltk.corpus.stopwords.words('portuguese')
        self.spacy_model_ = spacy.load("pt_core_news_lg")
        self.pca_model_ = PCA(n_components=2)
        self.pos_model_ = joblib.load(f"{os.path.dirname(__file__)}/POS_tagger_brill.pkl")
        self.negative_adverbs_ = ["não", "nem", "tampouco", "nunca", "jamais", "nada"]

    def fit(self, samples: List[dict]) -> None:
        tokens_list = [self._get_tokens_with_mask(sample) for sample in samples]
        self._fit_idf(tokens_list)
        self._fit_pca(samples)

    def get_features(self, samples: List[dict]) -> pd.DataFrame:
        features = []
        for sample in samples:

            new_features = {}
            for method_name in dir(self):
                if method_name == "_compute_principal_components":
                    principal_components = self._compute_principal_components(sample)
                    new_features["principal_component_0"] = principal_components[0]
                    new_features["principal_component_1"] = principal_components[1]
                elif method_name[:9] == "_compute_":
                    method = getattr(self, method_name)
                    new_features[method_name[9:]] = method(sample)
            features.append(new_features)

        return pd.DataFrame(features)

    @staticmethod
    def _get_tokens_with_mask(sample: dict) -> List[str]:
        tokens = [item for item in sample["tokens"]]
        tokens[sample["index_1"]] = "[mask]"
        tokens[sample["index_2"]] = "[mask]"
        return tokens

    def _fit_idf(self, tokens_list: List[list]) -> None:
        counter: dict = Counter()
        for tokens in tokens_list:
            for ngram in self._get_ngrams(tokens):
                counter[ngram] += 1
        self.idf_ = {}
        for ngram, count in counter.items():
            self.idf_[ngram] = math.log(float(len(tokens_list)) / count)

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int = 1) -> set:
        ngrams = set()
        for i in range(len(tokens)):
            if i+n <= len(tokens):
                words = tuple([word.lower() for word in tokens[i:i+n]])
                if "[mask]" not in words:
                    ngrams.add(words)
        return ngrams

    def _fit_pca(self, samples: List[dict]) -> None:
        x = []
        for sample in samples:
            before = self._get_sentence(sample, "before")
            between = self._get_sentence(sample, "between")
            after = self._get_sentence(sample, "after")
            x.append(self.spacy_model_(" ".join([before, between, after])).vector)
        self.pca_model_.fit(np.array(x))

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

    def _compute_n_stopwords_before_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "before")
        return len([token for token in tokens if token.lower() in self.stopwords_])

    def _compute_n_stopwords_between_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "between")
        return len([token for token in tokens if token.lower() in self.stopwords_])

    def _compute_n_stopwords_after_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "after")
        return len([token for token in tokens if token.lower() in self.stopwords_])

    def _compute_idf_before_entities(self, sample: dict) -> float:
        tokens = self._get_tokens(sample, "before")
        ngrams = self._get_ngrams(tokens)
        idfs = [self.idf_[item] for item in ngrams if item in self.idf_]
        return np.mean(idfs) if idfs else 0

    def _compute_idf_between_entities(self, sample: dict) -> float:
        tokens = self._get_tokens(sample, "between")
        ngrams = self._get_ngrams(tokens)
        idfs = [self.idf_[item] for item in ngrams if item in self.idf_]
        return np.mean(idfs) if idfs else 0

    def _compute_idf_after_entities(self, sample: dict) -> float:
        tokens = self._get_tokens(sample, "after")
        ngrams = self._get_ngrams(tokens)
        idfs = [self.idf_[item] for item in ngrams if item in self.idf_]
        return np.mean(idfs) if idfs else 0

    def _compute_principal_components(self, sample: dict) -> np.ndarray:
        before = self._get_sentence(sample, "before")
        between = self._get_sentence(sample, "between")
        after = self._get_sentence(sample, "after")
        doc = self.spacy_model_(" ".join([before, between, after]))
        return self.pca_model_.transform(doc.vector.reshape(1, -1))[0]

    def _compute_n_prop_names_before_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "before")
        tokens_tagged = self.pos_model_.tag(tokens)
        return self._count_tags(tokens_tagged, "NPROP")

    def _compute_n_prop_names_between_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "between")
        tokens_tagged = self.pos_model_.tag(tokens)
        return self._count_tags(tokens_tagged, "NPROP")

    def _compute_n_prop_names_after_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "after")
        tokens_tagged = self.pos_model_.tag(tokens)
        return self._count_tags(tokens_tagged, "NPROP")

    def _compute_n_pronouns_before_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "before")
        tokens_tagged = self.pos_model_.tag(tokens)
        return self._count_tags(tokens_tagged, "PROPESS")

    def _compute_n_pronouns_between_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "between")
        tokens_tagged = self.pos_model_.tag(tokens)
        return self._count_tags(tokens_tagged, "PROPESS")

    def _compute_n_pronouns_after_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "after")
        tokens_tagged = self.pos_model_.tag(tokens)
        return self._count_tags(tokens_tagged, "PROPESS")

    def _compute_n_neg_adverbs_before_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "before")
        return len([token for token in tokens if token.lower() in self.negative_adverbs_])

    def _compute_n_neg_adverbs_between_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "between")
        return len([token for token in tokens if token.lower() in self.negative_adverbs_])

    def _compute_n_neg_adverbs_after_entities(self, sample: dict) -> int:
        tokens = self._get_tokens(sample, "after")
        return len([token for token in tokens if token.lower() in self.negative_adverbs_])

    @staticmethod
    def _count_tags(tokens_tagged: List[tuple], search: str) -> int:
        count = 0
        previous_tag = ""
        for _, tag in tokens_tagged:
            if tag == search and previous_tag != search:
                count += 1
            previous_tag = tag
        return count

    def _get_sentence(self, sample: dict, location: str) -> str:
        tokens = self._get_tokens(sample, location)
        return " ".join(tokens)

    def _get_tokens(self, sample: dict, location: str) -> List[str]:
        indexes = self._get_sorted_indexes(sample)
        if location == "before":
            tokens = sample["tokens"][:indexes[0]]
        elif location == "between":
            tokens = sample["tokens"][indexes[0]+1:indexes[1]]
        elif location == "after":
            tokens = sample["tokens"][indexes[1]+1:]
        return tokens

    @staticmethod
    def _get_sorted_indexes(sample: dict) -> List[int]:
        return sorted([sample["index_1"], sample["index_2"]])
