import helper
import random


def printlist(db):
    ''' Print the list of problems in the database
    :param db: Mongodb database
    :return: A dictionary mapping the index in the printed list to the id of the problems in the db
    '''
    cursor = db.problems.find()
    eq = '=' * 70
    print eq
    print 'List of problems'
    i = 1
    problem_idx_to_id = dict()
    prior_begin = ' [prior: '
    prior_end = ']'
    posterior_begin = ' [posterior: '
    posterior_end = ']'
    for item in cursor:
        prior_str = prior_begin + str(item['prior']) + prior_end
        posterior_str = posterior_begin + str(item['posterior']) + posterior_end
        print str(i) + '. ' + item['name'] + prior_str + posterior_str
        problem_idx_to_id[i] = item['_id']
        i += 1
    print eq
    return problem_idx_to_id


def increment(db, problem_hash):
    ''' Increment the prior for this problem and set posterior equal to prior
    :param db: The Mongodb database
    :param problem: Hash value of the problem for which to increment the prior
    :return: None, update the db
    '''
    problem = db.problems.find_one({'hash': problem_hash})
    problem['prior'] += 1
    problem['posterior'] = problem['prior']
    db.problems.update({'_id': problem['_id']}, problem)


def sample(db, p):
    '''
    :param db: The Mongodb database
    :param p: A string that is either 'prior' or 'posterior' depending on what we want to sample from
    :return: A problem sampled from the problems database according to p
    '''
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
    ''' Query for a problem
    :param db: The Mongodb database
    :return: None
    '''
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
                helper.erroronezero()


def maxposterior(db):
    ''' Return the maximum (normalized) posterior value among all the problems in the database
    :param db: The Mongodb database
    :return: The maximum posterior probability among all the problems in the database
    '''
    cursor = db.problems.find()
    m = 0.0
    total = 0.0
    for item in cursor:
        total += item['posterior']
        if item['posterior'] > m:
            m = item['posterior']
    m /= total
    return m


def adjustposteriors(db, question, response, confidence=0.9):
    ''' Adjust the posterior of all the problems depending on the question, response and confidence
    :param db: The Mongodb database
    :param question: The question object
    :param response: Response (0 or 1)
    :param tolerance: Number in [0, 1] showing confidence in correctness of response
    :return: None, just update db entries
    '''
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


def thresholdset(db, t):
    '''
    :param db: The Mongodb database
    :param t: A parameter t
    :return: Return set of problem names whose posterior is > t
    '''
    s = set()
    cursor = db.problems.find()
    for item in cursor:
        if item['posterior'] > t:
            s.add(item['name'])
    return s


def printset(problemnames):
    ''' Print a set of problem names
    :param problemnames: Set of problem names
    :return: None, just print the set
    '''
    s = ', '.join(item for item in problemnames)
    print '{' + s + '}'


def delete(db, problem_id):
    ''' Delete problem from both the problems and question database
    :param db: The Mongodb database
    :param problem_id: The Mongodb database id of the problem
    :return: None, modify database in place
    '''
    problem_hash = db.problems.find_one({'_id': problem_id})['hash']
    db.problems.remove(problem_id)
    cursor = db.questions.find()
    for question in cursor:
        neg_problems = [x for x in question['negproblems'] if x != problem_hash]
        pos_problems = [x for x in question['posproblems'] if x != problem_hash]
        question['negproblems'] = neg_problems
        question['posproblems'] = pos_problems
        db.questions.update({'_id': question['_id']}, question)