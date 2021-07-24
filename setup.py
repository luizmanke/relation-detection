from setuptools import setup, find_packages
from relation_detection._version import __version__


setup(
    name="relation_detection",
    version=__version__,
    packages=find_packages(exclude=("tests"))
)
