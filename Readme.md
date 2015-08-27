# Ask Minmax!

Ask Minmax is an expert system for optimization problems targeted towards the non-expert user.

## How it works

### Initialization: 
   
 * An instance`expert` of the `Expert` class contains its own database as an attribute. 
 * The two main documents of an expert database are `problems` and `questions`. 
 * A problem in the `problems` document contains the following attributes 
    `{'name', 'hash', 'prior', 'posterior', 'posquestions', 'negquestions'}`.
 * A question in the `questions` document contains the following attirbutes
    `{'name', 'hash', 'prior', 'posterior', 'loglikelihood', posproblems', 'negproblems'}`.
 * When a new problem or a question is encountered, the program gets rid of all non-alpha characters,
    converts it to lower case and then computes an `md5` hash of it which is stored in the `hash` attribute.
 * The `prior` for a problem and a question is set to `1` the first time it is encountered with the prior
    being a count of how frequently a particular problem is encountered or a question is asked.
 * The `posterior` for a problem is intialized equal to its `prior`.
 
 
### Training

 * During the training phase the program samples a problem proportional to its `prior` and outputs it.
 * If the problem is incorrect, it asks for the correct problem and a separating question.
 * Separating questions come in two categories: *positive* and *negative*. 
 * A *positive* separating question is one where the answer is YES for the correct problem and NO for the
 wrongly guessed problem.
 * A *negative* separating question is one where the answer is NO for the correct problem and YES for the 
 wrongly guessed problem.
 * The program therefore learns when it makes a mistake!
 * Note that the `prior` for the correct problem is incremented during the training phase, thus your training
 problems should reflect the same bias as in the expected real world application.
 
### Flow of prediction
 
 * During the prediction phase the program samples a question according to its `posterior` and asks it to you.
 * It also asks you for a *confidence* number between 0 and 1 which indicates how confident you are about your 
 answer.
 * For a YES answer it updates the `posteriors` of the problems as follows. 
 * For every problem which for which the answer to this question is a NO, we reweigh as
 `posterior *= (1 - confidence)`. Therefore if you are a 100% confident in your answer, this problem is essentially
 out of running.
 * For every problem for which the answer to this question is a YES, we reweigh as 
 `posterior *= confidence`. Therefore if you are a 100% confident in your answer, this problem retains its full
  `prior` weight.
 * The `prior` for the correct problem is incremented. 
 * A similar operation is done for problems with a NO answer to this question.
 * The posteriors of every question in the database are now updated in the following manner.
 * Let `pos` be the total `posterior` mass of problems which have a YES answer to this question.
 * Let `neg` be the total `posterior` mass of problems which have a NO answer to this question.
 * The `loglikelihood` of this question is defined as `|log(pos) - log(neg)|`, which captures the discriminative
 power of this question with respect to the surviving problems.
 * The `posterior` of this question is then updated as `posterior *= exp(loglikelihood)`; in other words questions
 with higher discriminative power w.r.t to the surviving problems should have a higher `posterior`.
 * The question that was asked in this round gets its `posterior` set to 0. (Note that it may be asked again a few
 steps down the line).
 * The next question is then sampled according to the latest `posterior` distribution.
 
### Other stuff
 
 * You can store the database files as a BSON object (stored in `database/db`).
 * To see the code organization see [organization.md](src/askminmax/organization.md)
 * Snakefood visualization of the code [food.pdf](src/askminmax/food.pdf)
 * Comes with a small default [database](database/db)
 
## Prerequisites: 
 - A running [Mongodb](https://www.mongodb.org/) server 
 - [pymongo](https://pypi.python.org/pypi/pymongo/)
 - [nltk](http://www.nltk.org/)
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
