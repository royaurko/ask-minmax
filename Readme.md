# Ask Minmax!

Ask Minmax is an expert system for optimization problems targeted towards the non-expert user.

## How it works

The user either inputs a bunch of optimization problems 

```tex
\sum = \prod
```

## Prerequisites: 
 - A running [Mongodb](https://www.mongodb.org/) server 
 - Pymongo for python 2.7

## Quick Start:

To install it as a library first clone the repository

```shell
git clone git@github.com:royaurko/ask-minmax.git

```

The next step would be to `cd` to the directory you installed it in and use

```shell
sudo python setup.py install
```

To create an instance of the Expert system first import the `Expert` class and
then use the basic constructor

```python
from askminmax.expert import Expert
expert = Expert()
```

It will ask you to either import stuff from an existing database or it will give 
you the option to create one of your own. Finally to run the expert system use

```python
expert.run()
```
