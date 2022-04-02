import torch
from termcolor import colored


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    return device


def print_sentence(sample: dict) -> None:
    tokens = [token for token in sample["tokens"]]
    tokens[sample["index_1"]] = colored(tokens[sample["index_1"]], "blue", attrs=["bold"])
    tokens[sample["index_2"]] = colored(tokens[sample["index_2"]], "red", attrs=["bold"])
    sentence = " ".join(tokens)
    print(sentence)
