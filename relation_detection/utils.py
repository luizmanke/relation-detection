from termcolor import colored


def print_sentence(sample: dict) -> None:
    tokens = [token for token in sample["tokens"]]
    tokens[sample["index_1"]] = colored(tokens[sample["index_1"]], "blue", attrs=["bold"])
    tokens[sample["index_2"]] = colored(tokens[sample["index_2"]], "red", attrs=["bold"])
    sentence = " ".join(tokens)
    print(sentence)
