import nltk
import numpy as np
from lime.lime_text import LimeTextExplainer
from typing import Any, List


def explain_with_lime(model: Any, sample: dict) -> None:

    sample = dict(sample)

    # get original tokens
    sample["tokens"][sample["index_1"]] = "[E1]"
    sample["tokens"][sample["index_2"]] = "[E2]"
    original_sentence = " ".join(sample["tokens"])
    original_tokens = _tokenize_to_words(original_sentence)

    # get entities location
    for i, token in enumerate(original_tokens):
        if token == "[E1]":
            index_1 = i
        if token == "[E2]":
            index_2 = i

    # create sentence without entities
    sample["tokens"][sample["index_1"]] = ""
    sample["tokens"][sample["index_2"]] = ""
    sentence = " ".join(sample["tokens"])

    # define function
    def _predict(
            sentences: List[str],
            index_1: int = index_1,
            index_2: int = index_2
    ) -> np.ndarray:

        # Create samples
        samples = []
        for sentence in sentences:
            tokens = _tokenize_to_words(sentence)
            for index, element in sorted(zip([index_1, index_2], ["[E1]", "[E2]"])):
                tokens.insert(index, element)
            samples.append({
                "tokens": tokens,
                "index_1": index_1,
                "index_2": index_2
            })

        return model.predict(samples, return_proba=True, for_lime=True)

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
