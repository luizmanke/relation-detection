from setuptools import setup, find_packages


with open("requirements.txt") as req_file:
    dependencies = req_file.read().splitlines()

setup(
    name="relation_detection",
    packages=find_packages(exclude=("tests")),
    install_requires=dependencies
)
