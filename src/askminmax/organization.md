# Organization of the code

Below we list the main modules and the functions they contain with a short description of what they do. The
functions are written as `[return variable] function([parameters])`.

## [helper.py](helper.py)

* Depends on: `hashlib`, `re`
* Functions:
    - `None erroronzero()`: Error message
    - `None erroronnumber():` Error message
    - `s strip(s):` Strip leading spaces from `s` from left and right
    - `val gethashval(s):` Get hash vaulue of `s` after removing non-alpha chars from it
    - `m mass(db, table, hashval, property)`: Look up item with hashval from db.table and return mass of property
    
    
## [database.py](database.py)

* Depends on: `subprocess`, `helper`, `pymongo`
* Functions:
    - `client connect()`: Get the host and port number from the user and connect a client
    - `(client, db) initializedb()`: Connect client, drop all tables in the database and return
    - `None dumpdb(db)`: Dump database to BSON file
    - `db recoverdb(client)`: Recover db from BSON file
    
## [problems.py](problems.py)

* Depends on: `helper` and `random`
* Functions:
    - `problem_idx_to_id printlist(db)`: Print the list of problems with their priors & posteriors and return
    a dictionary item mapping the indexes in the printed list to their id in the database
    - `None increment(db, problem_hash)`: Increment the prior of the problem with hash value = `problem_hash` 
    and set its posterior equal to its prior
    - `problem sample(db, p)`: Sample a problem from the database according to the distribution specified by `p`
    where `p \in {'prior', 'posterior'}`
    - `None query(db)`: Query for a problem and enter it into the database
    - `m maxposterior(db)`: Return the maximum (normalized) posterior for a problem in the database
    - `None adjustposteriors(db, question, response, confidence)`: For a question and its response this adjusts the 
    posterior of problems
    - `s thresholdset(db, t)`: Return set of all problems with posteriors > t
    - `None printset(problemnames)`: Format and print the set of problem names in `problemnames` 
    - `None delete(db, problem_id)`: Delete the problem with Mongodb id `problem_id` from the questions and
    problems database
    
## [questions.py](questions.py)

* Depends on: `random`, `helper`, `math`
* Functions:
    - `question_idx_to_id printlist(db)`: Print the list of questions with their priors & posteriors and return
    a dictionary item mapping the indexes in the printed list to their id in the database
    - `None increment(db, question_hash)`: Increment the prior of the question with hash value = `question_hash` 
    and set its posterior equal to its prior 
    - `question sample(db, p)`: Sample a question from the database according to the distribution specified by `p`
    where `p \in {'prior', 'posterior'}`
    - `m maxposterior(db)`: Return the maximum (normalized) posterior for a question in the database
    - `None adjustposteriors(db)`: Refresh the posteriors and log-likelihood of the questions in the database
    - `None delete(db, question_id)`: Delete the question with Mongodb id `question_id` from the questions
    and problems database
    
## [sepquestions.py](sepquestions.py)

* Depends on: `helper`, `questions`, `problems`
* Functions: 
    - `(correct, correct_hash) getcorrectproblem(db)`: Query the correct problem from the user and return the
     name and the hash value
    - `None separatingquestion(db, problem)`: Ask a separating question between wrong problem and correct problem
    - `(correct_problem['name'], correct_problem['hash']) parseproblemlist(db)`: Helper function that takes user's
    input of correct problem into account and returns the name and database id of the problem
    - `[neg_qhash_val] parsenegsingle(db, problem, correct_hash)`: Parse a single negative separating question
    - `[pos_qhash_va] parsepossingle(db, problem, correct_hash)`: Parse a single positive separating question
    - `neg_qhash_list parseneglist(db, question_idx_to_id)`: Parse negative questions already in the database and
    return the list of their hash values
    - `pos_qhash_list parseposlist(db, question_idx_to_id)`: Parse positive questions already in the database and
    return the list of their hash values 
    - `None setlists(db, qhash, problem, correct, correct_hash, flag)`: Insert qhashval into the positive 
    and negative list (resp.) of the positive problem and the negative problem 
    



## [training.py](training.py)

* Depends on: `helper`, `problems`, `sepquestions`
* Functions: 
    - `None train(db)`: Query the number of times to train and call training
    - `None training(n, db)`: Train for `n` number of times

## [expert.py](expert.py)

* Depends on `helper`, `database`, `problems`, `questions`, `sepquestions`, `training`
* Implements the `Expert` class with the following methods:
    - `None __init__(self)`: Constructor for Expert class that initializes its database etc.
    - `None printtable(self)`: Print the problem and question tables together with their priors & posteriors
    - `None run(self)`: Controls the main program flow
    - `None resetposteriors(self)`: Reset the posteriors of the problems and questions
    - `None adjustposteriors(self, question, response, confidence)`: Calls the adjustposterior functions of
    problems and questions respectively taking into account confidence
    - `None askquestions(self, n)`: Ask `n` questions by sampling questions according to their posterior
    weights
    - `None predictsingle(self)`: Predict a single problem by sampling a problem according to its posterior
    - `None predictset(self, n)`: Predict a set of problems by sampling the `n` times
    - `None querybackup(self)`: Query whether to backup the database of the expert instance
    - `None controlprediction(self)`: Control the flow of prediction by directing calls to `predictsingle` or
    `predictset` as warranted
    - `None delete(self)`: Deletes problems and questions from the database by calling `problems.delete` and
    `questions.delete`
    - `None download(self, keywords)`: Download arxiv papers with keywords and store them in `db.papers`; 
    keywords are comma separated, for example `"TSP, maximum flow, minimum cut"`

## [arxiv.py](arxiv.py)

* Depends on `urllib2`, `feedparser`, `hashlib`, `re`, `os`, `time`
* Functions:
    - `cleaned_text clean(text)`: Clean
    - `scrunched_text scrunch(text)`: Scrunch
    - `None download(db, keywords)`: List of keywords, downloads papers and saves it in `db.papers`
    
    
## [natural_breaks.py](natural_breaks.py)

 * Depends on `jenks`, `numpy`
 * Functions:
    - `gvf gvf(array, classes`: Return the the goodness of fit value of Jenks run on array with `classes` number of 
    classes.
    - `num classify(value, breaks)`: Helper function for `gvf` 
    
    
## [clusters.py](cluster.py)

* Depends on `gensim`
* Functions:
    