import spacy
from termcolor import colored


def download_spacy_model() -> None:
    if not spacy.util.is_package("pt_core_news_lg"):
        print("\nDownloading spacy model...")
        spacy.cli.download("pt_core_news_lg", False, False, "--quiet")  # type: ignore
        print("")


def print_sentence(sample: dict) -> None:
    tokens = [token for token in sample["tokens"]]
    tokens[sample["index_1"]] = colored(tokens[sample["index_1"]], "blue", attrs=["bold"])
    tokens[sample["index_2"]] = colored(tokens[sample["index_2"]], "red", attrs=["bold"])
    sentence = " ".join(tokens)
    print(sentence)
