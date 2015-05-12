# Ask Minmax!

Ask Minmax is an expert system for optimization problems targeted to a non-technical audience.  

## Prerequisites: 
 - A running [Mongodb](https://www.mongodb.org/) server 
 - Pymongo for python 2.7

The main expert system file is in src/expert.py

## Quick Start:

Once mongodb is up and running simply clone the repo and run `expert.py`

 ```
 git clone git@github.com:royaurko/ask-minmax.git
 python src/expert.py

 ```

## Todo:
- compute frequency for setting priors instead of using uniform priors when it has been used enough
- make posteriors small instead of 0
- explore gensim, locality sensitive hashing to make it noise stable
- explore bbns 
