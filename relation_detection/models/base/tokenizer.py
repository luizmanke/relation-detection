from transformers import AutoTokenizer
from typing import List


class BaseTokenizer:

    def __init__(self, transformer_name: str) -> None:
        self.tokenizer_ = AutoTokenizer.from_pretrained(transformer_name)
        self.tokenizer_.add_tokens(["[E1]", "[E2]"])
        self.max_sequence_length_ = 512

    def transform(self, samples: List[dict], pad: bool = False) -> List[dict]:
        samples_tokenized = self._tokenize(samples, pad)
        samples_ids = self._tokens_to_ids(samples_tokenized)
        return samples_ids

    def _tokenize(self, samples: List[dict], pad: bool) -> List[dict]:
        EXTRA_CLS_TOKEN = 1
        samples_tokenized: List[dict] = []
        for sample in samples:

            tokens: List[str] = []
            for i, word in enumerate(sample["tokens"]):
                tokens_wordpiece = self.tokenizer_.tokenize(word)

                # add entity marker token
                if i == sample["index_1"]:
                    new_index_1 = len(tokens) + EXTRA_CLS_TOKEN
                    tokens.extend(["[E1]"])
                elif i == sample["index_2"]:
                    new_index_2 = len(tokens) + EXTRA_CLS_TOKEN
                    tokens.extend(["[E2]"])
                else:
                    tokens.extend(tokens_wordpiece)

            tokens = tokens[:self.max_sequence_length_ - 2]
            samples_tokenized.append({
                "tokens": tokens,
                "index_1": new_index_1,
                "index_2": new_index_2
            })

        if pad:
            max_length = max([len(sample["tokens"]) for sample in samples_tokenized])
            for sample in samples_tokenized:
                sample["tokens"] = (
                    sample["tokens"] + ["[PAD]"] * (max_length - len(sample["tokens"])))

        return samples_tokenized

    def _tokens_to_ids(self, samples: List[dict]) -> List[dict]:
        samples_ids: List[dict] = []
        for sample in samples:
            ids = self.tokenizer_.convert_tokens_to_ids(sample["tokens"])
            ids_with_special_tokens = self.tokenizer_.build_inputs_with_special_tokens(ids)
            samples_ids.append({
                "tokens": ids_with_special_tokens,
                "index_1": sample["index_1"],
                "index_2": sample["index_2"]
            })
        return samples_ids
