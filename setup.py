from setuptools import setup, find_packages


# get dependencies
with open("requirements.txt") as req_file:
    dependencies = req_file.read().splitlines()

# ignore dev dependencies
test_index = dependencies.index("# testing")
dependencies = dependencies[:test_index]

setup(
    name="relation_detection",
    packages=find_packages(exclude=("tests")),
    install_requires=dependencies
)
