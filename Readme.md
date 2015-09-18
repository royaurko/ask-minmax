# Ask Minmax!

Ask Minmax is an expert system for optimization problems targeted towards the non-expert user.


## How it works
 * Ask-minmax associates a **prior** to every problem and question in the database. 
 * The prior of a problem is proportional to the number of times there is a query intended for that particular
  problem. 
 * The prior of a question reflects the **gain in information** by asking the question  - in other words
 it is inversely proportional to the **expected conditional entropy** of the distribution of problem posteriors
 conditioned on the response to this question.
 * At every step a question is sampled proportional to it's posterior.
 * The posteriors of a problem are updated according to the **confidence** level in your answer.
 * The posteriors of a question are updated reflecting the information gain provided for this new distribution.
 * The algorithm outputs the most popular problems by doing a 1 dimensional k-means (a.k.a
[Jenks natural breaks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization) ).
 * A similar (clustering) idea is applied to questions, at every step only the "most useful questions" are
  sampled from.
 * The algorithm is greedy in the sense that it only cares about questions that reduce the entropy on the
 posterior distribution supported on the "most relevant" (coming from Jenks/k-means) problems.
 * You can visualizes the changing distribution in a simple matplotlib plot.
 
## Other stuff
 
 * You can store the database files as a BSON object (stored in `database/db`).
 * To see the code organization see [organization.md](src/askminmax/organization.md)
 * Comes with a small default [database](database/db)
 
## Prerequisites: 
The last three dependencies are due to Google's word2vec and are still a work in progress.

 * Python 2.7
 * A running [Mongodb](https://www.mongodb.org/) server 
 * [pymongo](https://pypi.python.org/pypi/pymongo/)
 * [jenks](https://github.com/perrygeo/jenks)
 * [scipy](http://www.scipy.org/)
 * [numpy](http://www.scipy.org/)
 * [matplotlib](http://matplotlib.org/)
 * [nltk](http://www.nltk.org/)
 * [scikit-learn](https://pypi.python.org/pypi/scikit-learn/0.16.1)
 * [gensim](https://pypi.python.org/pypi/gensim)

## Quickstart:

Install using `pip`.

```shell
pip install "git+https://github.com/royaurko/ask-minmax.git#egg=ask-minmax"
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

## Fun with Word2Vec

You can train your own word2vec model on your domain specific dataset. In this case we train it on
abstracts downloaded from `arxiv`, it has a vocabulary size of `175930` at the moment. Here's a fun
example in IPython:

```python
In [1]: from gensim.models import Word2Vec
In [2]: model = Word2Vec.load('model/model_175930')
In [3]: model.doesnt_match(['bipartite', 'non-bipartite', 'stable', 'matching', 'scheduling'])
Out[3]: 'scheduling'
```



