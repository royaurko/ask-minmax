#!/usr/bin/python
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="askminmax",
    version="1.0.0",
    author="Aurko Roy",
    author_email="aurko@gatech.edu",
    description=("A Bayesian expert system for optimization problems "),
    license="BSD",
    keywords="expert system",
    url="https://github.com/royaurko/ask-minmax",
    install_requires=['pymongo', 'jenks', 'scipy', 'numpy', 'matplotlib', 'nltk', 'scikit-learn', 'gensim'],
    packages=['askminmax'],
    package_dir={'askminmax': 'src/askminmax'},
    package_data={'askminmax': ['src/askminmax/database']},
    long_description=read('Readme.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
