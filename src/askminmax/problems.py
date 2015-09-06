import helper
import random
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt


def get_separating_questions(problem1, problem2):
    """ Print the separating questions already in our database between problem1 and problem2
    :param db: The Mongodb database
    :param problem1: The dictionary of problem #1
    :param problem2: The dictionary of problem #2
    :param question_idx_to_id: The dictionary mapping question indexes to Mongodb id's
    :return: Set of separating questions as a tuple
    """
    # First get questions with YES answer for hash1 and NO answer for hash2
    hash1_posquestions = set(problem1['posquestions'])
    hash1_negquestions = set(problem1['negquestions'])
    hash2_posquestions = set(problem2['posquestions'])
    hash2_negquestions = set(problem2['negquestions'])
    h1_pos_h2_neg = hash1_posquestions.intersection(hash2_negquestions)
    h2_pos_h1_neg = hash1_negquestions.intersection(hash2_posquestions)
    return h1_pos_h2_neg, h2_pos_h1_neg


def print_list(db):
    """ Print the list of problems in the database
    :param db: Mongodb database
    :return: A dictionary mapping the index in the printed list to the id of the problems in the db
    """
    cursor = db.problems.find()
    eq = '-' * 115
    print eq
    i = 1
    problem_idx_to_id = dict()
    template = "{Index:5} | {Name:70} | {Prior:15} | {Posterior:15}"
    print template.format(Index="Index", Name="Problem Name", Prior="Prior", Posterior="Posterior")
    print eq
    for item in cursor:
        d = {'Index': i, 'Name': item['name'], 'Prior': item['prior'], 'Posterior': item['posterior']}
        print template.format(**d)
        problem_idx_to_id[i] = item['_id']
        i += 1
    print eq
    return problem_idx_to_id


def increment(db, problem_hash, n=1):
    """ Increment the prior for this problem and set posterior equal to prior
    :param db: The Mongodb database
    :param problem: Hash value of the problem for which to increment the prior
    :param n: Increment by n
    :return: None, update the db
    """
    problem = db.problems.find_one({'hash': problem_hash})
    problem['prior'] += n
    problem['posterior'] = problem['prior']
    db.problems.update({'_id': problem['_id']}, problem)


def sample(db, p):
    """
    :param db: The Mongodb database
    :param p: A string that is either 'prior' or 'posterior' depending on what we want to sample from
    :return: A problem sampled from the problems database according to p
    """
    cursor = db.problems.find()
    count = cursor.count()
    if count < 1:
        '''Trying to sample from empty collection'''
        print 'Empty problem set!'
        query(db)
    weight = 0
    for item in cursor:
        weight += item[p]
    r = random.uniform(0, weight)
    s = 0.0
    cursor = db.problems.find()
    for item in cursor:
        s += item[p]
        if r < s:
            return item
    return item


def query(db):
    """ Query for a problem
    :param db: The Mongodb database
    :return: None
    """
    response = 1
    while response:
        pname = helper.strip(raw_input('Problem name: '))
        # tokens = helper.gettokens(pname)
        hashval = helper.gethashval(pname)
        item = db.problems.find_one({'hash': hashval})
        if item is None:
            # It is a new item, create the problem dictionary and insert into data base
            prior = 1
            posterior = 1
            posquestions = list()
            negquestions = list()
            d = {'name': pname, 'hash': hashval, 'prior': prior,
                 'posterior': posterior, 'posquestions': posquestions,
                 'negquestions': negquestions}
            db.problems.insert(d)
        while True:
            try:
                response = int(raw_input('Continue (0/1)? '))
                break
            except ValueError:
                helper.error_one_zero()


def max_posterior(db):
    """ Return the maximum (normalized) posterior value among all the problems in the database
    :param db: The Mongodb database
    :return: The maximum posterior probability among all the problems in the database
    """
    cursor = db.problems.find()
    m = 0.0
    total = 0.0
    for item in cursor:
        total += item['posterior']
        if item['posterior'] > m:
            m = item['posterior']
    m /= total
    return m


def adjust_posteriors(db, question, response, confidence=0.9):
    """ Adjust the posterior of all the problems depending on the question, response and confidence
    :param db: The Mongodb database
    :param question: The question object
    :param response: Response (0 or 1)
    :param tolerance: Number in [0, 1] showing confidence in correctness of response
    :return: None, just update db entries
    """
    cursor = db.problems.find()
    for problem in cursor:
        if response and question['hash'] in problem['negquestions']:
            problem['posterior'] *= 1.0 - confidence
        if response and question['hash'] in problem['posquestions']:
            problem['posterior'] *= confidence
        if not response and question['hash'] in problem['posquestions']:
            problem['posterior'] *= 1.0 - confidence
        if not response and question['hash'] in problem['negquestions']:
            problem['posterior'] *= confidence
        db.problems.update({'_id': problem['_id']}, problem)


def threshold_set(db, t):
    """
    :param db: The Mongodb database
    :param t: A parameter t
    :return: Return set of problem names whose posterior is > t
    """
    s = set()
    cursor = db.problems.find()
    for item in cursor:
        if item['posterior'] > t:
            s.add(item['name'])
    return s


def print_set(problem_names):
    """ Print a set of problem names
    :param problem_names: Set of problem names
    :return: None, just print the set
    """
    s = ', '.join(item for item in problem_names)
    print '{' + s + '}'


def delete(db, problem_id):
    """ Delete problem from both the problems and question database
    :param db: The Mongodb database
    :param problem_id: The Mongodb database id of the problem
    :return: None, modify database in place
    """
    problem_hash = db.problems.find_one({'_id': problem_id})['hash']
    db.problems.remove(problem_id)
    cursor = db.questions.find()
    for question in cursor:
        neg_problems = [x for x in question['negproblems'] if x != problem_hash]
        pos_problems = [x for x in question['posproblems'] if x != problem_hash]
        question['negproblems'] = neg_problems
        question['posproblems'] = pos_problems
        db.questions.update({'_id': question['_id']}, question)


def get_entropy(db, most_likely_problems=list()):
    """ Get the entropy of the posteriors
    :param db: The Mongodb database
    :param most_likely_problems: Optional argument, if provided confine only to this set
    :return: The entropy of the posterior distribution of the problems
    """
    p = np.array([])
    cursor = db.problems.find()
    most_likely_problems_hash = set([item['hash'] for item in most_likely_problems])
    for problem in cursor:
        if most_likely_problems_hash:
            if problem['hash'] in most_likely_problems_hash:
                p = np.append(p, problem['posterior'])
        else:
            p = np.append(p, problem['posterior'])
    return entropy(p)


def plot_posteriors(db):
    """ Plot the posteriors of the problems in the database
    :param db: The Mongodb database
    :return: None, plot the distribution of the posteriors of the problems
    """
    p = np.array([])
    cursor = db.problems.find()
    for problem in cursor:
        p = np.append(p, problem['posterior'])
    s = p.sum()
    n = p.shape
    for i in xrange(n[0]):
        p[i] /= s
    x = xrange(len(p))
    plt.ion()
    plt.clf()
    plt.plot(x, p, 'bo', x, p, 'k')
    plt.xlabel('Problems')
    plt.ylabel('Posteriors')
    plt.show()