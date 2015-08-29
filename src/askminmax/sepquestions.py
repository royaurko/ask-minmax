import helper
import questions
import problems


def getcorrectproblem(db):
    ''' Query the correct problem from the user and return the name and the hash value
    :param db: The Mongodb database
    :return: Name and hash value
    '''
    while True:
        try:
            response = int(raw_input('Correct problem in the list (0/1)? '))
            break
        except ValueError:
            helper.erroronezero()
    if response:
        # Correct problem is already in our list
        correct, correct_hash = parseproblemlist(db)
        print 'Correct problem: ' + correct
    else:
        # Correct problem is not in our list, get its hash value
        correct = helper.strip(raw_input('What is the correct problem?\n'))
        correct_hash = helper.gethashval(correct)
    return (correct, correct_hash)


def separatingquestion(db, problem, correct, correct_hash):
    ''' Ask a separating question between wrong problem and correct problem
    :param db: The Mongodb database
    :param problem: The problem dictionary for which we want the separating question
    :param correct: The name of the correct problem
    :param correct_hash: The hash value of the correct problem
    :return: None
    '''
    question_idx_to_id = questions.printlist(db)
    while True:
        try:
            s = 'Separating question for ' + problem['name']
            s = s + ' in the list (0/1)? '
            response = int(raw_input(s))
            break
        except ValueError:
            helper.erroronezero()
    if response:
        # Separating question is already in our list, get their hash values
        pos_qhash_list = parseposlist(db, question_idx_to_id)
        neg_qhash_list = parseneglist(db, question_idx_to_id)
    else:
        # Separating question is not in our list ask for one
        pos_qhash_list = parsepossingle(db, problem, correct_hash)
        neg_qhash_list = parsenegsingle(db, problem, correct_hash)
    for qhash in pos_qhash_list:
        # For every question set its list correctly and increment its prior and set posterior equal to prior
        setlists(db, qhash, problem, correct, correct_hash, True)
        questions.increment(db, qhash)
    for qhash in neg_qhash_list:
        # For every question set its list correctly and increment its prior and set posterior equal to prior
        setlists(db, qhash, problem, correct, correct_hash, False)
        questions.increment(db, qhash)


def parseproblemlist(db):
    ''' Parse list of problems in the database
    :param db: The Mongodb database
    :return: The name and hash value of the correct problem
    '''
    problem_idx_to_id = problems.printlist(db)
    while True:
        try:
            idx = int(raw_input('Enter correct problem number: '))
            if idx in problem_idx_to_id:
                break
            else:
                print 'Please enter valid problem number!'
        except ValueError:
            helper.errornumber()
    # Get correct problem id
    correct_id = problem_idx_to_id[idx]
    correct_problem = db.problems.find_one({'_id': correct_id})
    return (correct_problem['name'], correct_problem['hash'])

def parsenegsingle(db, problem, correct_hash):
    ''' Parse a negative single question
    :param db: The Mongodb database
    :param problem: The dictionary item for the wrong problem
    :param correct_hash: Hash value of correct problem
    :return: List of hash value of the negative question
    '''
    neg_qname = helper.strip(raw_input('Enter a new negative separating question?\n'))
    if not neg_qname:
        return []
    neg_qhashval = helper.gethashval(neg_qname)
    neg_question = db.questions.find_one({'hash': neg_qhashval})
    if neg_question is None and neg_qname:
        # This is a new negative question, insert into db with prior initialized to 1
        posproblems = [problem['hash']]
        negproblems = [correct_hash]
        prior = 1
        posterior = prior
        loglikelihood = 0.0
        d = {'name': neg_qname, 'hash': neg_qhashval, 'prior': prior,
             'posterior': posterior, 'posproblems': posproblems,
             'negproblems': negproblems, 'loglikelihood': loglikelihood}
        db.questions.insert_one(d)
    return [neg_qhashval]

def parsepossingle(db, problem, correct_hash):
    ''' Parse a negative single question
    :param db: The Mongodb database
    :param problem: The dictionary item for the wrong problem
    :param ctokens: Hash value of correct problem
    :return: List of hash value of the positive question
    '''
    pos_qname = helper.strip(raw_input('Enter a new positive separating question?\n'))
    if not pos_qname:
        # User did not enter a positive question
        return []
    pos_qhashval = helper.gethashval(pos_qname)
    pos_question = db.questions.find_one({'hash': pos_qhashval})
    if pos_question is None and pos_qname:
        # This is a new positive question, insert into db with prior initialized to 1
        posproblems = [correct_hash]
        negproblems = [problem['hash']]
        prior = 1
        posterior = prior
        loglikelihood = 0.0
        d = {'name': pos_qname, 'hash': pos_qhashval, 'prior': prior,
             'posterior': posterior, 'posproblems': posproblems,
             'negproblems': negproblems, 'loglikelihood': loglikelihood}
        db.questions.insert_one(d)
    return [pos_qhashval]

def parseneglist(db, question_idx_to_id):
    ''' Parse negative questions already in the list
    :param db: The Mongodb database
    :param question_idx_to_id: The dictionary that maps question numbers to Mongodb ids
    :return: List of hash values of negative questions
    '''
    neg_list = raw_input('Enter negative question numbers separated by spaces: ')
    neg_list = map(int, neg_list.strip().split())
    neg_qid_list = [question_idx_to_id[x] for x in neg_list]
    neg_qhash_list = list()
    for item in neg_qid_list:
        hashval = db.questions.find_one({'_id': item})['hash']
        neg_qhash_list.append(hashval)
    return neg_qhash_list

def parseposlist(db, question_idx_to_id):
    '''  Get list of hash values of separating questions already in the list
    :param db: The Mongodb database
    :param question_idx_to_id: The dictionary that maps question numbers to Mongodb ids
    :return: List of hash values of positive questions
    '''
    pos_list = raw_input('Enter positive question numbers separated by spaces: ')
    pos_list = map(int, pos_list.strip().split())
    pos_qid_list = [question_idx_to_id[x] for x in pos_list]
    pos_qhash_list = list()
    for item in pos_qid_list:
        hashval = db.questions.find_one({'_id': item})['hash']
        pos_qhash_list.append(hashval)
    return pos_qhash_list

def setlists(db, qhash, problem, correct, correct_hash, flag):
    ''' Modify the positive and negative lists of the correct and wrong problem
    :param db: The Mongodb database
    :param qhash: The hash value of the question
    :param problem: The dictionary for the wrongly guessed problem
    :param correct: The name of the correct problem
    :param correct_hash: The hash value of the correct problem
    :param flag: Indicates whether it is a positive or a negative separating question
    :return: None, modify the database
    '''
    correct_problem = db.problems.find_one({'hash': correct_hash})
    if flag:
        clistname = 'posquestions'
        plistname = 'negquestions'
    else:
        clistname = 'negquestions'
        plistname = 'posquestions'
    if correct_problem is not None:
        # Add question hash value to appropriate list of the problem
        l = correct_problem[clistname]
        if qhash not in l:
            l.append(qhash)
            db.problems.update({'_id': correct_problem['_id']}, {
                '$set': {clistname: l}
            })
    else:
        # It is a new problem, populate its values
        prior = 1
        posterior = prior
        if flag:
            posquestions = [qhash]
            negquestions = list()
        else:
            posquestions = list()
            negquestions = [qhash]
        d = {'name': correct, 'hash': correct_hash, 'prior': prior,
             'posterior': posterior, 'posquestions': posquestions,
             'negquestions': negquestions}
        db.problems.insert_one(d)
    plist = problem[plistname]
    if qhash not in plist:
        plist.append(qhash)
        db.problems.update_one({'_id': problem['_id']}, {
            '$set': {plistname: plist}
        })