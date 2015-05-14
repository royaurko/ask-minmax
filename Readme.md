# Ask Minmax!

Ask Minmax is an expert system for optimization problems targeted towards the non-expert user.
When run for the first time it queries the user whether to build a new database of problems.
Ideally when deployed the user should simply use the database provided with the source code,
the name of the database is (you guessed it) `database`. 

If a new database is being built then it asks the user for a list of problems and then in the training
phase guesses a random problem. If the answer is wrong it queries the user for a separating question.
Separating questions come in two flavors - **positive** and **negative**. A *positive separating question* is
true for the correct problem but false for the guessed problem. A *negative separating question* is
false for the correct problem but true for the guessed problem. Clearly these are the only two scenarios
under which the program can learn anything non-trivial.

While running in prediction mode, you have the option to stop the program at any point and make it
output either a single problem or a set of most likely problems. You get to choose the upper limit
on the size of this set and the program only outputs problems that have not been ruled out yet (aka
non-zero posterior). 

Right now the algorithm uses a simple Bayesian update rule to learn. More sophisticated approaches 
to learning are also currently being considered. 

## Prerequisites: 
 - A running [Mongodb](https://www.mongodb.org/) server 
 - Pymongo for python 2.7

The main expert system file is in `src/expert.py`

## Quick Start:

Once mongodb is up and running simply clone the repo and run `expert.py`

 ```
 git clone git@github.com:royaurko/ask-minmax.git
 python src/expert.py

 ```
