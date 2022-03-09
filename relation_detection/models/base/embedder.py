import numpy as np
import spacy


class Embedder:

    def __init__(self) -> None:
        self.embedder_ = spacy.load("pt_core_news_lg")

    def transform(self, sentences: list) -> np.ndarray:
        vectors_list = []
        for sentence in sentences:
            sentence = sentence if sentence else " "
            vectors_list.append(self.embedder_(sentence).vector)
        vectors = np.array(vectors_list)
        return vectors
