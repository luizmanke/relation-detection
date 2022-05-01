from setuptools import setup, find_packages
from setuptools.command.install import install


class PostInstallCommand(install):
    def run(self):
        install.run(self)

        import nltk
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)


DEPENDENCIES = [
    "imbalanced-learn==0.8.0",
    "lime==0.2.0.1",
    "matplotlib==3.5.1",
    "nltk==3.6.2",
    "numpy==1.21.0",
    "pandas==1.3.0",
    "pt_core_news_lg @  https://github.com/explosion/spacy-models/releases/download/pt_core_news_lg-3.1.0/pt_core_news_lg-3.1.0.tar.gz",
    "seaborn==0.11.2",
    "scikit-learn==1.0.2",
    "spacy==3.1.0",
    "termcolor==1.1.0",
    "torch==1.9.0",
    "transformers==4.8.2",
    "types-termcolor==1.1.1"
]

setup(
    name="relation_detection",
    packages=find_packages(exclude=("tests", "notebooks")),
    install_requires=DEPENDENCIES,
    package_data={"relation_detection": ["*.pkl"]},
    cmdclass={"install": PostInstallCommand}
)
