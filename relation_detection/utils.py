import nltk


def download_nltk_model():
    try:
        nltk.data.find("tokenizers/punkt")
    except IndexError:
        print("\nDownloading nltk model...")
        nltk.download("punkt")
        print("")
