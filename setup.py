#!/usr/bin/python
import os
from setuptools import setup
import sys


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

data_dir = os.path.join(sys.prefix, "local/lib/python2.7/dist-package/askmminmax")

setup(
    name="askminmax",
    version="1.0.0",
    author="Aurko Roy",
    author_email="aurko@gatech.edu",
    description=("A Bayesian expert system for optimization problems "),
    license="BSD",
    keywords="expert system",
    url="https://github.com/royaurko/ask-minmax",
    install_requires=['pymongo', 'scipy', 'numpy', 'matplotlib', 'nltk', 'sklearn', 'gensim', 'feedparser', 'jenks'],
    dependency_links=['git+https://github.com/perrygeo/jenks.git#egg=jenks'],
    packages=['askminmax'],
    package_dir={'askminmax': 'src/askminmax'},
    data_files = [('askminmax', [os.path.join(data_dir, 'database'), os.path.join(data_dir, 'model')])],
    long_description=read('Readme.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
