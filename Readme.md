# Ask Minmax!

Ask Minmax is an expert system for optimization problems targeted towards the non-expert user.


## How it works
Ask-minmax associates a prior to every problem and question in the database. The prior of a problem is roughly 
proportional to the number of times there was a query intended for that particular problem. The prior of a question
reflects the relative usefulness of the question - i.e. the total "current weight" of problems it can separate. 
The posteriors are initialized equal to the priors. At every step, a question is sampled proportional to a prior
and depending on its answer the posteriors of the problems and questions are updated. One has the option to use
[Jenks natural breaks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization) algorithm 
with a desired *Goodness of fit* parameter to output a set of most likely problems. The algorithm also queries
whether you want to refine the set further by asking more questions. 
 
## Other stuff
 
 * You can store the database files as a BSON object (stored in `database/db`).
 * To see the code organization see [organization.md](src/askminmax/organization.md)
 * Snakefood visualization of the code [food.pdf](src/askminmax/food.pdf)
 * Comes with a small default [database](database/db)
 
## Prerequisites: 
 - A running [Mongodb](https://www.mongodb.org/) server 
 - [pymongo](https://pypi.python.org/pypi/pymongo/)
 - [nltk](http://www.nltk.org/)
 - [jenks](https://github.com/perrygeo/jenks)
 - [scikit-learn](https://pypi.python.org/pypi/scikit-learn/0.16.1)
 - [gensim](https://pypi.python.org/pypi/gensim)

## Quick Start:

Clone the repository and install it as a library using 

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
