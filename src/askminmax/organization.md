# Organization of the code

Below we list the main modules and the functions they contain with a short description of what they do. The
functions are written as `[return variable] function([parameters])`.

## [helper.py](helper.py)

* Depends on: `hashlib`, `re`
* Functions:
    - 'None error_spaces()`: Error message
    - `None error_one_zero()`: Error message
    - `None error_number():` Error message
    - `None error_key():` Error message
    - `s strip(s):` Strip leading spaces from `s` from left and right
    - `val get_hash(s):` Get hash value of `s` after removing non-alpha chars from it
    - `m mass(db, table, hash_value, property_name)`: Look up item with hash_value
     from db.table and return mass of property
    
    
## [database.py](database.py)

* Depends on: `subprocess`, `helper`, `pymongo`
* Functions:
    - `client connect()`: Get the host and port number from the user and connect a client
    - `(client, db) initialize_db()`: Connect client, drop all tables in the database and return
    - `None dump_db(db)`: Dump database to BSON file
    - `db recover_db(client)`: Recover db from BSON file
    
## [problems.py](problems.py)

* Depends on: `helper` , `random`, `scipy.stats`, `numpy`, `matplotlib.pyplot`
* Functions:
    - `(h1_pos_h2_neg, h2_pos_h1_neg) get_separating_questions(problem1, problem2):`  Return the separating questions 
    already in our database between problem1 and problem2
    - `problem_idx_to_id print_list(db)`: Print the list of problems with their priors & posteriors and return
    a dictionary item mapping the indexes in the printed list to their id in the database
    - `None increment(db, problem_hash)`: Increment the prior of the problem with hash value = `problem_hash` 
    and set its posterior equal to its prior
    - `problem sample(db, p)`: Sample a problem from the database according to the distribution specified by `p`
    where `p \in {'prior', 'posterior'}`
    - `None query(db)`: Query for a problem and enter it into the database
    - `m max_posterior(db)`: Return the maximum (normalized) posterior for a problem in the database
    - `None adjust_posteriors(db, question, response, confidence=0.9)`: For a question and its response 
    this adjusts the posterior of problems
    - `s threshold_set(db, t)`: Return set of all problems with posteriors > t
    - `None print_set(problemnames)`: Format and print the set of problem names in `problemnames` 
    - `None delete(db, problem_id)`: Delete the problem with Mongodb id `problem_id` from the questions and
    problems database
    - `entropy get_entropy(db, most_likely_problems=list()):` Get the entropy of the posterior distribution on
    problems, if most_likely_problems is not empty then confine to this as support
    - `None plot_posteriors(db):` Plot the posteriors in interactive mode so one can see how the distribution
    changes over time
    
## [questions.py](questions.py)

* Depends on: `random`, `scipy.stats`, `numpy`, `problems`
* Functions:
    - `question_idx_to_id print_list(db)`: Print the list of questions with their priors & posteriors and return
    a dictionary item mapping the indexes in the printed list to their id in the database
    - `None increment(db, question_hash)`: Increment the prior of the question with hash value = `question_hash` 
    and set its posterior equal to its prior 
    - `question sample(db, p)`: Sample a question from the database according to the distribution specified by `p`
    where `p \in {'prior', 'posterior'}`
    - `m max_posterior(db):` Return the maximum (normalized) posterior for a question in the database
    - `None reset_priors(db):` Reset the priors of the questions in the database to reflect how much 
    it can bring entropy down
    - `None adjust_posteriors(db)`: Update the posteriors of the questions
    - `None delete(db, question_id)`: Delete the question with Mongodb id `question_id` from the questions
    and problems database
    - `entropy conditional_entropy(db, q, response, most_likely_problems=list()):` Return entropy of the 
    posterior distribution of problems conditioned on response to question, if most_likely_problems is not
    empty then confine to this as support
    
## [sepquestions.py](sepquestions.py)

* Depends on: `helper`, `questions`, `problems`
* Functions: 
    - `(correct, correct_hash) get_correct_problem(db)`: Query the correct problem from the user and return the
     name and the hash value
    - `None display_separating_questions(db, wrong, correct, question_idx_to_id)`: Display the list of separating 
    questions already in the database
    - `None ask_separating_question(db, wrong, correct):` Ask a separating question between wrong problem
     and correct problem
    - `correct get_problem_from_list(db):` Ask the user for the correct problem and return its dictionary
    - `[neg_question_hash_value] parse_negative_single_question(db, wrong, correct)`: Parse a single negative 
    separating question given dictionary for wrong and correct problem
    - `[pos_question_hash_value] parse_positive_single_question(db, wrong, correct)`: Parse a single positive 
    separating question given dictionary for wrong and correct problem
    - `negative_question_hash_list parse_negative_list_questions(db, wrong, correct, question_idx_to_id):` Parse
     negative questions already in the database and return the list of their hash values
    - `positive_question_hash_list parse_positive_list_questions(db, wrong, correct, question_idx_to_id):` Parse
     positive questions already in the database and return the list of their hash values 
    - `None set_problem_lists(db, question_hash, wrong, correct, flag):` Modify the positive and negative lists
     of the correct and wrong problem



## [training.py](training.py)

* Depends on: `helper`, `problems`, `sepquestions`
* Functions: 
    - `None train(db)`: Query the number of times to train and call training
    - `None training(n, db)`: Train for `n` number of times

## [expert.py](expert.py)

* Depends on `helper`, `database`, `problems`, `questions`, `sepquestions`, `training`
* Implements the `Expert` class with the following methods:
    - `None __init__(self):` Constructor for Expert class that initializes its database etc.
    - `None train(self):` Call the train subroutine from training to learn separating questions
    - `None delete(self):` Allows user to delete a problem or a question from the database
    - `None print_table(self):` Print the problem and question tables together with their priors & posteriors
    - `None run(self)`: Controls the main program flow
    - `None reset_posteriors(self):` Reset the posteriors of the problems and questions
    - `None adjust_question_posteriors(self, responses_known_so_far, most_likely_problems):` Adjust the posteriors 
    of the questions
    - `None adjust_problem_posteriors(self, question, response, confidence):` Adjust the posteriors of 
    problems
    - `None ask_questions(self, most_likely_questions)`: Ask a question and return the question, 
    response and confidence level
    - `None predict_single(self):` Predict a single problem by sampling a problem according to its posterior
    - `None predict_set(self, n):` Predict a set of problems by sampling the `n` times
    - `None get_feedback(self, most_likely):` Query the user if the correct problem was in this set
    - `None query_backup(self):` Query whether to backup the database of the expert instance
    - `most_likely_questions query_gvf_question(self):` Query goodness of fit value for the questions from user
    - `most_likely_problems query_gvf_problems(self):` Query goodness of fit value for the problems from user
    - `None control_prediction(self)`: Control flow of questions
    - `most_likely fit_posteriors(self, document, desired_gvf=0.8):` Cluster the posteriors using Jenks Natural Breaks
    - `None add_problem(self):` Add a problem to the database and query for YES questions and NO questions
    - `None add_question(self):` Add a question to the database and query for problems with YES answers and NO answers
    - `None delete(self)`: Deletes problems and questions from the database by calling `problems.delete` and
    `questions.delete`
    - `None download(self, flag, keywords)`: Download arxiv papers with keywords and store them in `db.papers`;    
    keywords are comma separated, for example `"TSP, maximum flow, minimum cut"`
    - `count count_papers(self):` Count the number of papers in `db.papers`
    - `None make_uniform(self):` Make the priors of all the problems uniform
    - `None cluster(self):` Run k-means on the word2vec vectors
    
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
    