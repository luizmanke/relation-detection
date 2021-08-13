import nltk
import spacy


def download_nltk_model():
    try:
        nltk.data.find("corpora/stopwords")
        nltk.data.find("tokenizers/punkt")
    except Exception:
        print("\nDownloading nltk model...")
        nltk.download("stopwords")
        nltk.download("punkt")
        print("")


def download_spacy_model():
    if not spacy.util.is_package("pt_core_news_lg"):
        print("\nDownloading spacy model...")
        spacy.cli.download("pt_core_news_lg", False, False, "--quiet")  # type: ignore
        print("")
