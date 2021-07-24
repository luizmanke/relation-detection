from setuptools import setup, find_packages
from relation_detection._version import __version__


setup(
    name="relation_detection",
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description="long_description",
    long_description_content_type="text/markdown",
    url="https://github.com/luizmanke/relation-detection",
    project_urls={
        "Bug Tracker": "https://github.com/luizmanke/relation-detection/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "relation_detection"},
    packages=find_packages(),
    python_requires=">=3.6",
)
