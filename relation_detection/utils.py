import copy
import nltk
import numpy as np
from lime.lime_text import LimeTextExplainer
from typing import Any, List


def explain_with_lime(model: Any, sample: dict) -> None:

    sample = copy.deepcopy(sample)

    sample["tokens"][sample["index_1"]] = "[E1]"
    sample["tokens"][sample["index_2"]] = "[E2]"
    original_sentence = " ".join(sample["tokens"])
    original_tokens = _tokenize_to_words(original_sentence)
    index_1, index_2 = None, None
    for i, token in enumerate(original_tokens):
        if token == "[E1]":
            index_1 = i
        if token == "[E2]":
            index_2 = i
    assert index_1 is not None and index_2 is not None

    sample["tokens"][sample["index_1"]] = ""
    sample["tokens"][sample["index_2"]] = ""
    sentence = " ".join(sample["tokens"])

    def _predict(sentences: List[str], index_1=index_1, index_2=index_2) -> np.ndarray:

        sentences_reordered = []
        for sentence in sentences:
            tokens = _tokenize_to_words(sentence)
            for index, element in sorted(zip([index_1, index_2], ["[E1]", "[E2]"])):
                tokens.insert(index, element)
            sentences_reordered.append(" ".join(tokens))
        
        tokens_without_pad = [
            model.model_.tokenizer_.tokenize(sentence) for sentence in sentences_reordered]

        max_length = len(tokens_without_pad[0])
        tokens_with_pad = [
            tokens + ["[PAD]"] * (max_length - len(tokens)) for tokens in tokens_without_pad]

        ids = []
        for tokens in tokens_with_pad:
            tokens = tokens[:model.model_.max_sequence_length_ - 2]
            ids.append(model.model_.tokenizer_.build_inputs_with_special_tokens(
                model.model_.tokenizer_.convert_tokens_to_ids(tokens)
            ))

        samples_tokenized = []
        for id in ids:
            a, b = None, None
            for i, token in enumerate(id):
                if token == 29794:
                    a = i + 1
                if token == 29794:
                    b = i + 1
            assert a is not None and b is not None
            samples_tokenized.append({
                "tokens": id,
                "index_1": a,
                "index_2": b
            })

        return model.model_._predict_proba(samples_tokenized)

    explainer = LimeTextExplainer(
        class_names=["not related", "related"],
        split_expression=_tokenize_to_words,
        mask_string="[PAD]",
        bow=False
    )

    lime_values = explainer.explain_instance(
        text_instance=sentence,
        classifier_fn=_predict,
        num_samples=10,
        num_features=len(_tokenize_to_words(sentence))
    )

    lime_values.show_in_notebook(text=True, labels=(lime_values.available_labels()[0],))


def _tokenize_to_words(sentence: str) -> List[str]:

    # download
    try:
        nltk.data.find("tokenizers/punkt")
    except IndexError:
        print("\nDownloading nltk model...")
        nltk.download("punkt")
        print("")

    # tokenize
    i = 0
    tokens = []
    KNOWN_TOKENS = ["E1", "E2", "PAD"]
    raw_tokens = nltk.word_tokenize(sentence, language="portuguese")
    while i < len(raw_tokens):
        if raw_tokens[i] == "[" and raw_tokens[i+1] in KNOWN_TOKENS and raw_tokens[i+2] == "]":
            tokens.append(f"[{raw_tokens[i+1]}]")
            i += 3
        else:
            tokens.append(raw_tokens[i])
            i += 1

    return tokens
