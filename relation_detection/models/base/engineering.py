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

    NEGATIVE_ADVERBS = ["não", "nem", "tampouco", "nunca", "jamais", "nada"]

    def __init__(self, use_numerical: bool = True, use_categorical: bool = False) -> None:
        download_nltk_model()
        download_spacy_model()
        self.stopwords_ = nltk.corpus.stopwords.words('portuguese')
        self.spacy_model_ = spacy.load("pt_core_news_lg")
        self.pca_model_ = PCA(n_components=2)
        self.pos_model_ = joblib.load(f"{os.path.dirname(__file__)}/POS_tagger_brill.pkl")
        self.use_numerical_ = use_numerical
        self.use_categorical_ = use_categorical

    def fit(self, samples: List[dict]) -> None:
        tokens_list = [self._get_tokens_with_mask(sample) for sample in samples]
        if self.use_numerical_:
            self._fit_idf(tokens_list)
            self._fit_pca(samples)
        if self.use_categorical_:
            self._fit_pos(tokens_list)

    def transform(self, samples: List[dict]) -> pd.DataFrame:
        features: List[dict] = []
        for sample in samples:

            new_features: dict = {}
            for method_name in dir(self):
                if (
                    (method_name[:11] == "_numerical_" and self.use_numerical_) or
                    (method_name[:13] == "_categorical_" and self.use_categorical_)
                ):
                    method = getattr(self, method_name)
                    method(sample, new_features)
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

    def _fit_pos(self, tokens_list: List[list]) -> None:

        pos_set = set()
        for tokens in tokens_list:
            tokens_tagged = self.pos_model_.tag(tokens)
            new_pos = [x[1] for x in tokens_tagged]
            for pos in new_pos:
                pos_set.add(pos)

        self.pos_dict_ = {"NONE": 0, "ANY": 1, "UNKNOWN": 2}
        for i, pos in enumerate(pos_set):
            self.pos_dict_[pos] = i + 3

    def _numerical_length_before_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "before")
        input_dict["length_before_entities"] = len(sentence)

    def _numerical_length_between_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "between")
        input_dict["length_between_entities"] = len(sentence)

    def _numerical_length_after_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "after")
        input_dict["length_after_entities"] = len(sentence)

    def _numerical_n_tokens_before_entities(self, sample: dict, input_dict: dict) -> None:
        indexes = self._get_sorted_indexes(sample)
        input_dict["n_tokens_before_entities"] = indexes[0]

    def _numerical_n_tokens_between_entities(self, sample: dict, input_dict: dict) -> None:
        input_dict["n_tokens_between_entities"] = abs(sample["index_1"] - sample["index_2"]) - 1

    def _numerical_n_tokens_after_entities(self, sample: dict, input_dict: dict) -> None:
        indexes = self._get_sorted_indexes(sample)
        input_dict["n_tokens_after_entities"] = len(sample["tokens"]) - indexes[1] - 1

    def _numerical_entities_order(self, sample: dict, input_dict: dict) -> None:
        input_dict["entities_order"] = 0 if sample["index_1"] > sample["index_2"] else 1

    def _numerical_n_commas_before_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "before")
        input_dict["n_commas_before_entities"] = len(re.findall(",", sentence))

    def _numerical_n_commas_between_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "between")
        input_dict["n_commas_between_entities"] = len(re.findall(",", sentence))

    def _numerical_n_commas_after_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "after")
        input_dict["n_commas_after_entities"] = len(re.findall(",", sentence))

    def _numerical_n_semicolons_before_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "before")
        input_dict["n_semicolons_before_entities"] = len(re.findall(";", sentence))

    def _numerical_n_semicolons_between_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "between")
        input_dict["n_semicolons_between_entities"] = len(re.findall(";", sentence))

    def _numerical_n_semicolons_after_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "after")
        input_dict["n_semicolons_after_entities"] = len(re.findall(";", sentence))

    def _numerical_n_colons_before_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "before")
        input_dict["n_colons_before_entities"] = len(re.findall(":", sentence))

    def _numerical_n_colons_between_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "between")
        input_dict["n_colons_between_entities"] = len(re.findall(":", sentence))

    def _numerical_n_colons_after_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "after")
        input_dict["n_colons_after_entities"] = len(re.findall(":", sentence))

    def _numerical_n_periods_before_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "before")
        input_dict["n_periods_before_entities"] = len(re.findall(r"\.", sentence))

    def _numerical_n_periods_between_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "between")
        input_dict["n_periods_between_entities"] = len(re.findall(r"\.", sentence))

    def _numerical_n_periods_after_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "after")
        input_dict["n_periods_after_entities"] = len(re.findall(r"\.", sentence))

    def _numerical_n_exclamation_points_before_entities(
            self,
            sample: dict,
            input_dict: dict
    ) -> None:
        sentence = self._get_sentence(sample, "before")
        input_dict["n_exclamation_points_before_entities"] = len(re.findall("!", sentence))

    def _numerical_n_exclamation_points_between_entities(
            self,
            sample: dict,
            input_dict: dict
    ) -> None:
        sentence = self._get_sentence(sample, "between")
        input_dict["n_exclamation_points_between_entities"] = len(re.findall("!", sentence))

    def _numerical_n_exclamation_points_after_entities(
            self,
            sample: dict,
            input_dict: dict
    ) -> None:
        sentence = self._get_sentence(sample, "after")
        input_dict["n_exclamation_points_after_entities"] = len(re.findall("!", sentence))

    def _numerical_n_question_marks_before_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "before")
        input_dict["n_question_marks_before_entities"] = len(re.findall(r"\?", sentence))

    def _numerical_n_question_marks_between_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "between")
        input_dict["n_question_marks_between_entities"] = len(re.findall(r"\?", sentence))

    def _numerical_n_question_marks_after_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "after")
        input_dict["n_question_marks_after_entities"] = len(re.findall(r"\?", sentence))

    def _numerical_n_quotation_marks_before_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "before")
        input_dict["n_quotation_marks_before_entities"] = len(re.findall("'", sentence))

    def _numerical_n_quotation_marks_between_entities(
            self,
            sample: dict,
            input_dict: dict
    ) -> None:
        sentence = self._get_sentence(sample, "between")
        input_dict["n_quotation_marks_between_entities"] = len(re.findall("'", sentence))

    def _numerical_n_quotation_marks_after_entities(self, sample: dict, input_dict: dict) -> None:
        sentence = self._get_sentence(sample, "after")
        input_dict["n_quotation_marks_after_entities"] = len(re.findall("'", sentence))

    def _numerical_n_stopwords_before_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "before")
        input_dict["n_stopwords_before_entities"] = \
            len([token for token in tokens if token.lower() in self.stopwords_])

    def _numerical_n_stopwords_between_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "between")
        input_dict["n_stopwords_between_entities"] = \
            len([token for token in tokens if token.lower() in self.stopwords_])

    def _numerical_n_stopwords_after_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "after")
        input_dict["n_stopwords_after_entities"] = \
            len([token for token in tokens if token.lower() in self.stopwords_])

    def _numerical_idf_before_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "before")
        ngrams = self._get_ngrams(tokens)
        idfs = [self.idf_[item] for item in ngrams if item in self.idf_]
        input_dict["idf_before_entities"] = np.mean(idfs) if idfs else 0

    def _numerical_idf_between_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "between")
        ngrams = self._get_ngrams(tokens)
        idfs = [self.idf_[item] for item in ngrams if item in self.idf_]
        input_dict["idf_between_entities"] = np.mean(idfs) if idfs else 0

    def _numerical_idf_after_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "after")
        ngrams = self._get_ngrams(tokens)
        idfs = [self.idf_[item] for item in ngrams if item in self.idf_]
        input_dict["idf_after_entities"] = np.mean(idfs) if idfs else 0

    def _numerical_principal_components(self, sample: dict, input_dict: dict) -> None:
        before = self._get_sentence(sample, "before")
        between = self._get_sentence(sample, "between")
        after = self._get_sentence(sample, "after")
        doc = self.spacy_model_(" ".join([before, between, after]))
        principal_components = self.pca_model_.transform(doc.vector.reshape(1, -1))[0]
        input_dict["principal_component_0"] = principal_components[0]
        input_dict["principal_component_1"] = principal_components[1]

    def _numerical_n_prop_names_before_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "before")
        tokens_tagged = self.pos_model_.tag(tokens)
        input_dict["n_prop_names_before_entities"] = self._count_tags(tokens_tagged, "NPROP")

    def _numerical_n_prop_names_between_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "between")
        tokens_tagged = self.pos_model_.tag(tokens)
        input_dict["n_prop_names_between_entities"] = self._count_tags(tokens_tagged, "NPROP")

    def _numerical_n_prop_names_after_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "after")
        tokens_tagged = self.pos_model_.tag(tokens)
        input_dict["n_prop_names_after_entities"] = self._count_tags(tokens_tagged, "NPROP")

    def _numerical_n_pronouns_before_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "before")
        tokens_tagged = self.pos_model_.tag(tokens)
        input_dict["n_pronouns_before_entities"] = self._count_tags(tokens_tagged, "PROPESS")

    def _numerical_n_pronouns_between_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "between")
        tokens_tagged = self.pos_model_.tag(tokens)
        input_dict["n_pronouns_between_entities"] = self._count_tags(tokens_tagged, "PROPESS")

    def _numerical_n_pronouns_after_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "after")
        tokens_tagged = self.pos_model_.tag(tokens)
        input_dict["n_pronouns_after_entities"] = self._count_tags(tokens_tagged, "PROPESS")

    def _numerical_n_neg_adverbs_before_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "before")
        input_dict["n_neg_adverbs_before_entities"] = \
            len([token for token in tokens if token.lower() in self.NEGATIVE_ADVERBS])

    def _numerical_n_neg_adverbs_between_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "between")
        input_dict["n_neg_adverbs_between_entities"] = \
            len([token for token in tokens if token.lower() in self.NEGATIVE_ADVERBS])

    def _numerical_n_neg_adverbs_after_entities(self, sample: dict, input_dict: dict) -> None:
        tokens = self._get_tokens(sample, "after")
        input_dict["n_neg_adverbs_after_entities"] = \
            len([token for token in tokens if token.lower() in self.NEGATIVE_ADVERBS])

    def _categorical_pos_before_entity_1(self, sample: dict, input_dict: dict) -> None:
        self._compute_pos(sample, input_dict, "before", True, "pos_before_entity_1")

    def _categorical_pos_after_entity_1(self, sample: dict, input_dict: dict) -> None:
        self._compute_pos(sample, input_dict, "between", False, "pos_after_entity_1")

    def _categorical_pos_before_entity_2(self, sample: dict, input_dict: dict) -> None:
        self._compute_pos(sample, input_dict, "between", True, "pos_before_entity_2")

    def _categorical_pos_after_entity_2(self, sample: dict, input_dict: dict) -> None:
        self._compute_pos(sample, input_dict, "after", False, "pos_after_entity_2")

    def _compute_pos(
            self,
            sample: dict,
            input_dict: dict,
            location: str,
            before_entity: bool,
            feature_name: str
    ) -> None:
        N_POS = 3
        pos = ["NONE"] * N_POS
        tokens = self._get_tokens(sample, location)
        tokens_tagged = self.pos_model_.tag(tokens)
        tokens_tagged = tokens_tagged[-N_POS:] if before_entity else tokens_tagged[:N_POS]
        if tokens_tagged:
            pos[-len(tokens_tagged):] = [x[1] for x in tokens_tagged]
        for i, tag in enumerate(pos):
            key = tag if tag in self.pos_dict_ else "UNKNOWN"
            input_dict[f"{feature_name}_{i}"] = self.pos_dict_[key]

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
