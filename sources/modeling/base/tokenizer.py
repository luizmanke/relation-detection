from transformers import AutoTokenizer
from typing import List


class BaseTokenizer:

    def __init__(self, transformer_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
        self.max_sequence_length = 512

    def transform(self, samples: List[dict]) -> List[dict]:
        new_samples = []
        for sample in samples:
            new_sample = self._tokenize(
                sample['tokens'],
                sample['index_1'],
                sample['index_2']
            )
            new_samples.append(new_sample)
        return new_samples

    def _tokenize(self, words: List[str], index_1: int, index_2: int) -> dict:
        tokens: List[str] = []
        for i, word in enumerate(words):
            tokens_wordpiece = self.tokenizer.tokenize(word)

            # add entity marker token
            if i == index_1:
                new_index_1 = len(tokens) + 1
                tokens.extend(['[E1]'] + tokens_wordpiece + ['[/E1]'])
            elif i == index_2:
                new_index_2 = len(tokens) + 1
                tokens.extend(['[E2]'] + tokens_wordpiece + ['[/E2]'])
            else:
                tokens.extend(tokens_wordpiece)

        tokens = tokens[:self.max_sequence_length - 2]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(ids)

        return {
            "tokens": input_ids,
            "index_1": new_index_1,
            "index_2": new_index_2
        }
