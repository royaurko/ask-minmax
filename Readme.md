# Expert system for optimization problems

This is an ongoing project on building an expert system for optimization problems targeted towards
the novice user. 

Prerequisites: 
 - A running [Mongodb](https://www.mongodb.org/) server 

The main expert system file is in src/expert.py

Todo:
- compute frequency for setting priors instead of using uniform priors when it has been used enough
- make posteriors small instead of 0
- explore gensim, locality sensitive hashing to make it noise stable
- explore bbns 
