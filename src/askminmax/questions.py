import random
import helper
import math


def printlist(db):
    ''' Print the list of questions in the database
    :param db: The Mongodb database
    :return: A dictionary mapping the index in the printed list to the id of the questions in the db
    '''
    cursor = db.questions.find()
    eq = '=' * 70
    print eq
    print 'List of questions'
    i = 1
    question_idx_to_id = dict()
    prior_begin = ' [prior: '
    prior_end = ']'
    posterior_begin = ' [posterior: '
    posterior_end = ']'
    for item in cursor:
        prior_str = prior_begin + str(item['prior']) + prior_end
        posterior_str = posterior_begin + str(item['posterior']) + posterior_end
        print str(i) + '. ' + item['name'] + prior_str + posterior_str
        question_idx_to_id[i] = item['_id']
        i += 1
    print eq
    return question_idx_to_id

def increment(db, question_hash):
    ''' Increment the prior for this problem and set posterior equal to prior
    :param db: The Mongodb database
    :param question: The hash value for the question whose prior to increment
    :return: None, update the db by incrementing prior and set posterior = prior
    '''
    question = db.questions.find_one({'hash': question_hash})
    question['prior'] += 1
    question['posterior'] = question['prior']
    db.questions.update({'_id': question['_id']}, question)

def sample(db, p):
    ''' Sample a question from the database according to its p-value
    :param db: The mongodb database
    :param p: A string that is either 'prior' or 'posterior'
    :return: Dictionary of sampled question
    '''
    cursor = db.questions.find()
    count = cursor.count()
    zero_count = db.questions.find({p: 0}).count()
    if count < 1 or count == zero_count:
        # Trying to sample from empty question set or all the p-values are 0
        print 'No questions with non-zero ' + p + ' !'
        return
    weight = 0.0
    for item in cursor:
        weight += item[p]
    r = random.uniform(0, weight)
    s = 0.0
    cursor = db.questions.find()
    for item in cursor:
        s += item[p]
        if r < s:
            return item
    return item

def maxposterior(db):
    ''' Return the value of the maximum posterior among questions
    :param db: The Mongodb database
    :return: The maximum (normalized) posterior of a question in the database
    '''
    cursor = db.questions.find()
    m = 0.0
    total = 0.0
    for item in cursor:
        total += item['posterior']
        m = max(m, item['posterior'])
    if total:
        m /= total
    return m

def adjustposteriors(db):
    ''' Update the log likelihood and posterior of the questions
    :param db: The Mongodb database
    :return: None, update db in place
    '''
    cursor = db.questions.find()
    for q in cursor:
        if q['posterior'] > 0:
            table = 'problems'
            property = 'posterior'
            q_posproblem_mass = map(lambda x: helper.mass(db, table, x, property), q['posproblems'])
            q_posproblem_total_mass = reduce(lambda x, y: x + y, q_posproblem_mass)
            q_negproblem_mass = map(lambda x: helper.mass(db, table, x, property), q['negproblems'])
            q_negproblem_total_mass = reduce(lambda x, y: x + y, q_negproblem_mass)
            if q_posproblem_total_mass and q_negproblem_total_mass:
                pos = float(q_posproblem_total_mass)
                neg = float(q_negproblem_total_mass)
                q['loglikelihood'] = abs(math.log(pos) - math.log(neg))
                q['posterior'] *= math.exp(-q['loglikelihood'])
            else:
                # It does not appear as a separating question in any problem
                q['posterior'] = 0
            db.questions.update({'_id': q['_id']}, q)


def delete(db, question_id):
    ''' Delete question from both problems and questions database
    :param db: The Mongodb database
    :param question_id: The Mongodb id of the question to be deleted
    :return: None, modify database in place
    '''
    question_hash = db.questions.find_one({'_id': question_id})['hash']
    db.questions.remove(question_id)
    cursor = db.problems.find()
    for problem in cursor:
        neg_questions = [x for x in problem['negquestions'] if x != question_hash]
        pos_questions = [x for x in problem['posquestions'] if x != question_hash]
        problem['negquestions'] = neg_questions
        problem['posquestions'] = pos_questions
        db.problems.update({'_id': problem['_id']}, problem)