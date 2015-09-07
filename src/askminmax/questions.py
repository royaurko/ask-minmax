from __future__ import print_function
import random
from scipy.stats import entropy
import numpy as np
import problems


def print_list(db):
    """ Print the list of questions in the database
    :param db: The Mongodb database
    :return: A dictionary mapping the index in the printed list to the id of the questions in the db
    """
    cursor = db.questions.find()
    eq = '-' * 115
    print(eq)
    i = 1
    question_idx_to_id = dict()
    template = "{Index:5} | {Name:70} | {Prior:15} | {Posterior:15}"
    print(template.format(Index="Index", Name="Question Name", Prior="Prior", Posterior="Posterior"))
    print(eq)
    for item in cursor:
        d = {'Index': i, 'Name': item['name'], 'Prior': item['prior'], 'Posterior': item['posterior']}
        print(template.format(**d))
        question_idx_to_id[i] = item['_id']
        i += 1
    print(eq)
    return question_idx_to_id


def increment(db, question_hash, n=1):
    """ Increment the prior for this question and set posterior equal to prior
    :param db: The Mongodb database
    :param question: The hash value for the question whose prior to increment
    :param n: Increment by n
    :return: None, update the db by incrementing prior and set posterior = prior
    """
    question = db.questions.find_one({'hash': question_hash})
    question['prior'] += n
    question['posterior'] = question['prior']
    db.questions.update({'_id': question['_id']}, question)


def sample(db, p, most_likely_question_hash=set()):
    """ Sample a question from the database according to its p-value
    :param db: The mongodb database
    :param p: A string that is either 'prior' or 'posterior'
    :param most_likely_question_hash: A set of hash values of the questions to sample from
    :return: Dictionary of sampled question
    """
    cursor = db.questions.find()
    count = cursor.count()
    zero_count = db.questions.find({p: 0}).count()
    if count < 1 or count == zero_count:
        # Trying to sample from empty question set or all the p-values are 0
        print('No questions with non-zero ' + p + ' !')
        return
    weight = 0.0
    if most_likely_question_hash:
        for question in cursor:
            if question['hash'] in most_likely_question_hash:
                weight += question[p]
        r = random.uniform(0, weight)
        s = 0.0
        cursor = db.questions.find()
        for question in cursor:
            if question['hash'] in most_likely_question_hash:
                s += question[p]
                if r < s:
                    return question
        return question
    else:
        for question in cursor:
            weight += question[p]
        r = random.uniform(0, weight)
        s = 0.0
        cursor = db.questions.find()
        for question in cursor:
            s += question[p]
            if r < s:
                return question
        return question


def max_posterior(db):
    """ Return the value of the maximum posterior among questions
    :param db: The Mongodb database
    :return: The maximum (normalized) posterior of a question in the database
    """
    cursor = db.questions.find()
    m = 0.0
    total = 0.0
    for item in cursor:
        total += item['posterior']
        m = max(m, item['posterior'])
    if total:
        m /= total
    return m


def reset_priors(db):
    """ Reset the priors of the questions in the database to reflect how much it can bring entropy down
    :param db: Mongodb database
    :return: None, update db in place
    """
    cursor = db.questions.find()
    old_entropy = problems.get_entropy(db)
    for q in cursor:
        c_entropy = 0.5*conditional_entropy(db, q, True) + 0.5*conditional_entropy(db, q, False)
        q['prior'] = old_entropy - c_entropy
        q['posterior'] = q['prior']
        db.questions.update({'_id': q['_id']}, q)


def adjust_posteriors(db, responses_known_so_far, most_likely_problems):
    """ Update the posteriors of the questions
    :param db: The database
    :param responses_known_so_far: The responses known so far
    :param most_likely_problems: The most likely set of problems
    :return: None, update db in place
    """
    cursor = db.questions.find()
    old_entropy = problems.get_entropy(db, most_likely_problems)
    for q in cursor:
        # Update the posterior of a problem to reflect how much it brings down the entropy
        if q['hash'] in responses_known_so_far:
            # If a question was asked already
            q['posterior'] = 0.0
        else:
            # Question hasn't been asked yet, so assume either response is equally likely
            c_entropy = 0.5*conditional_entropy(db, q, True, most_likely_problems) \
                + 0.5*conditional_entropy(db, q, False, most_likely_problems)
            q['posterior'] = old_entropy - c_entropy
        db.questions.update({'_id': q['_id']}, q)


def delete(db, question_id):
    """ Delete question from both problems and questions database
    :param db: The Mongodb database
    :param question_id: The Mongodb id of the question to be deleted
    :return: None, modify database in place
    """
    question_hash = db.questions.find_one({'_id': question_id})['hash']
    db.questions.remove(question_id)
    cursor = db.problems.find()
    for problem in cursor:
        neg_questions = [x for x in problem['negquestions'] if x != question_hash]
        pos_questions = [x for x in problem['posquestions'] if x != question_hash]
        problem['negquestions'] = neg_questions
        problem['posquestions'] = pos_questions
        db.problems.update({'_id': problem['_id']}, problem)


def conditional_entropy(db, q, response, most_likely_problems=list()):
    """ The conditional entropy H(posterior | response to q)
    :param db: The Mongodb database
    :param q: The question
    :param response: The response to the question
    :param most_likely_problems: List of the most likely problems dictionary
    :return: The conditional entropy H(posterior | response to q)
    """
    posteriors = np.array([])
    cursor = db.problems.find()
    most_likely_problems_hash = set([item['hash'] for item in most_likely_problems])
    if response:
        for problem in cursor:
            if most_likely_problems_hash:
                if problem['hash'] not in most_likely_problems_hash:
                    continue
            if problem['hash'] in q['negproblems']:
                posteriors = np.append(posteriors, 0.0)
            else:
                posteriors = np.append(posteriors, problem['posterior'])
    else:
        for problem in cursor:
            if most_likely_problems_hash:
                if problem['hash'] not in most_likely_problems_hash:
                    continue
            if problem['hash'] in q['posproblems']:
                posteriors = np.append(posteriors, 0.0)
            else:
                posteriors = np.append(posteriors, problem['posterior'])
    if np.any(posteriors):
        return entropy(posteriors)
    return 0.0


def print_set(question_names):
    """ Print a set of question names
    :param question_names: Set of question names
    :return: None, just print the set
    """
    s = ', '.join(item for item in question_names)
    print('{' + s + '}')
