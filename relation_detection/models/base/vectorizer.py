import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer


class BaseVectorizer:

    def __init__(self, vectorizer_name: str) -> None:
        if vectorizer_name == "count":
            self.vectorizer = CountVectorizer()
        elif vectorizer_name == "spacy":
            if not spacy.util.is_package("pt_core_news_lg"):
                print("\nDownloading spacy model...")
                spacy.cli.download("pt_core_news_lg", False, False, "--quiet")  # type: ignore
                print("")
            self.vectorizer = spacy.load("pt_core_news_lg")
        self.vectorizer_name = vectorizer_name

    def fit(self, sentences: list) -> None:
        if self.vectorizer_name == "count":
            self.vectorizer.fit(sentences)
        elif self.vectorizer_name == "spacy":
            pass

    def transform(self, sentences: list) -> np.ndarray:
        if self.vectorizer_name == "count":
            vectors = self.vectorizer.transform(sentences).toarray()
        elif self.vectorizer_name == "spacy":
            vectors = []
            for sentence in sentences:
                vectors.append(self.vectorizer(sentence).vector)
            vectors = np.array(vectors)
        return vectors
