# Ask Minmax!

Ask Minmax is an expert system for optimization problems targeted towards the non-expert user.


## How it works
 * Ask-minmax associates a **prior** to every problem and question in the database. 
 * The prior of a problem is proportional to the number of times there is a query intended for that particular
  problem. 
 * At the beginning of every iteration, it asks the user for a human readable [summary](summaries.md).
 * (Document vectors)[https://cs.stanford.edu/~quocle/paragraph_vector.pdf] have been trained on 
  abstracts downloaded from arxiv and google-scholar.
 * A softmax classifier is built on top of these document vectors to output a probability distribution on
 the problems.
 * The prior thus reflects this probability distribution together with the frequency. 
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
 
## Prerequisites: 

 * Python 3
 * A running [Mongodb](https://www.mongodb.org/) server 
 * [pymongo](https://pypi.python.org/pypi/pymongo/)
 * [jenks](https://github.com/perrygeo/jenks)
 * [scipy](http://www.scipy.org/)
 * [numpy](http://www.scipy.org/)
 * [matplotlib](http://matplotlib.org/)
 * [nltk](http://www.nltk.org/)
 * [scikit-learn 0.17](https://pypi.python.org/pypi/scikit-learn/0.17)
 * [gensim](https://pypi.python.org/pypi/gensim)

## Quickstart:

Clone the repository and use

```shell
sudo python3 setup.py install
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
expert.run(data_set, doc2vec_model_path, classifier_path)
```
where `data_set` is the path to the data set (for e.g. [dataset](dataset)), 
 `doc2vec_model_path` is the path to the Doc2Vec model (for e.g. 
 [model_2015-11-25_08-44-27.d2v](model_2015-11-25_08-44-27.d2v)), and 
 `classifier_path` is the path to the softmax classifier 
 (for e.g. [model_2015-11-28_21-50-40.log](model_2015-11-28_21-50-40.log)).

## Doc2Vec model examples

You can train your own doc2vec model on your domain specific dataset. In this case we train it on
abstracts downloaded from `arxiv` and `google-scholar`. Here are some interesting examples in IPython:

```python
In [1]: from gensim.models import Doc2Vec
In [2]: model = Doc2Vec.load('model/[Insert model name]')
In [3]: model.doesnt_match(['bipartite', 'non-bipartite', 'stable', 'matching', 'scheduling'])
Out[3]: 'scheduling'
In [4]: model.doesnt_match(['facility location', 'minimum cut', 'maximum cut', 'sparsest cut'])
Out[4]: 'facility location'
In [5]: model.doesnt_match(['scheduling', 'routing', 'maximum', 'TSP', 'facility location'])
Out[5]: 'maximum'
```



