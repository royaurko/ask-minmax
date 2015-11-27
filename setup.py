#!/usr/bin/env/python
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="askminmax",
    version="1.0.0",
    author="Aurko Roy",
    author_email="aurko@gatech.edu",
    description="Expert system for optimization problems ",
    license="BSD",
    keywords="expert system",
    url="https://github.com/royaurko/ask-minmax",
<<<<<<< HEAD
    install_requires=['pymongo', 'scipy', 'numpy', 'matplotlib', 'nltk', 'scikit-learn', 'gensim', 'feedparser',
=======
    install_requires=['pymongo', 'scipy', 'numpy', 'matplotlib', 'nltk', 'scikit-learn==0.17', 'gensim', 'feedparser',
>>>>>>> 203d0ecc0999ed9641a61ac934a661116305b4b7
                      'jenks', 'lxml', 'cython', 'requests'],
    dependency_links=['git+https://github.com/perrygeo/jenks.git#egg=jenks'],
    packages=['askminmax'],
    package_dir={'askminmax': 'src/askminmax'},
    package_data={'askminmax': ['src/askminmax/database', 'src/askminmax/models']},
    long_description=read('Readme.md'),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
