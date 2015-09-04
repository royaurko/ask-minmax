import random
from scipy.stats import entropy
import numpy as np


def print_list(db):
    """ Print the list of questions in the database
    :param db: The Mongodb database
    :return: A dictionary mapping the index in the printed list to the id of the questions in the db
    """
    cursor = db.questions.find()
    eq = '-' * 115
    print eq
    i = 1
    question_idx_to_id = dict()
    template = "{Index:5} | {Name:70} | {Prior:15} | {Posterior:15}"
    print template.format(Index="Index", Name="Question Name", Prior="Prior", Posterior="Posterior")
    print eq
    for item in cursor:
        d = {'Index': i, 'Name': item['name'], 'Prior': item['prior'], 'Posterior': item['posterior']}
        print template.format(**d)
        question_idx_to_id[i] = item['_id']
        i += 1
    print eq
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
        print 'No questions with non-zero ' + p + ' !'
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
    for q in cursor:
        q['prior'] = change_in_entropy(db, q)
        q['posterior'] = q['prior']
        db.questions.update({'_id': q['_id']}, q)


def adjust_posteriors(db):
    """ Update the log likelihood and posterior of the questions
    :param db: The Mongodb database
    :return: None, update db in place
    """
    cursor = db.questions.find()
    for q in cursor:
        # Update the posterior of a problem to reflect how much it brings down the entropy
        q['posterior'] = change_in_entropy(db, q)
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


def change_in_entropy(db, q):
    """ The expected change in entropy when you ask this question, assuming each answer is equally likely
    :param db: The Mongodb database
    :param q: The dictionary of the question
    :return: The expected change in entropy
    """
    cursor = db.problems.find()
    old_pvalues, yes_pvalues, no_pvalues = np.array([]), np.array([]), np.array([])
    # Get the old_pvalues
    for problem in cursor:
        old_pvalues = np.append(old_pvalues, problem['posterior'])
    # Compute the old entropy
    old_entropy = entropy(old_pvalues)
    # Change in entropy if we answer YES to this question
    cursor = db.problems.find()
    for problem in cursor:
        if problem['hash'] in q['negproblems']:
            yes_pvalues = np.append(yes_pvalues, 0.0)
        else:
            yes_pvalues = np.append(yes_pvalues, problem['posterior'])
    # Compute the YES entropy
    yes_entropy = entropy(yes_pvalues)
    # Change in entropy if we answer NO to this question
    cursor = db.problems.find()
    for problem in cursor:
        if problem['hash'] in q['posproblems']:
            no_pvalues = np.append(no_pvalues, 0.0)
        else:
            no_pvalues = np.append(no_pvalues, problem['posterior'])
    # Compute the NO entropy
    no_entropy = entropy(no_pvalues)
    return old_entropy - 0.5*(yes_entropy + no_entropy)


def print_set(question_names):
    """ Print a set of question names
    :param question_names: Set of question names
    :return: None, just print the set
    """
    s = ', '.join(item for item in question_names)
    print '{' + s + '}'
