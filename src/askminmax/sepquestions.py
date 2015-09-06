from __future__ import print_function
import helper
import questions
import problems


def getcorrectproblem(db):
    """ Query the correct problem from the user and return its dictionary
    :param db: The Mongodb database
    :return: The dictionary item for the correct problem
    """
    while True:
        try:
            response = int(raw_input('Correct problem in the DB (0/1)? '))
            break
        except ValueError:
            helper.error_one_zero()
    if response:
        # Correct problem is in our database; get it's dictionary item
        correct = get_problem_from_list(db)
        print('Correct problem: ' + correct['name'])
    else:
        # Correct problem is not in our database
        correct_name = helper.strip(raw_input('What is the correct problem?\n'))
        correct_hash = helper.gethashval(correct_name)
        # Create a blank dictionary item for this new problem
        correct = {'name': correct_name, 'hash': correct_hash,
                   'prior': 1, 'posterior': 1, 'posquestions': [], 'negquestions': []}
    return correct


def display_separating_questions(db, wrong, correct, question_idx_to_id):
    """ Display the list of separating questions already in the database
    :param wrong: The dictionary of the wrong problem
    :param correct: The dictionary of the correct problem
    :param question_idx_to_id: Dictionary mapping question indices to mongodb ids
    :return: None
    """
    question_id_to_idx = {v: k for k, v in question_idx_to_id.items()}
    # Get the list of separating questions
    neg_q, pos_q = problems.get_separating_questions(wrong, correct)
    list1, list2  = list(), list()
    for qhash in pos_q:
        question = db.questions.find_one({'hash': qhash})
        list1.append(question_id_to_idx[question['_id']])
    for qhash in neg_q:
        question = db.questions.find_one({'hash': qhash})
        list2.append(question_id_to_idx[question['_id']])
    print('YES for ' + correct['name'] + ' NO for ' + wrong['name'] + ': ')
    print(list1)
    print('NO for ' + correct['name'] + ' YES for ' + wrong['name'] + ': ')
    print(list2)


def ask_separating_question(db, wrong, correct):
    """ Ask a separating question between wrong problem and correct problem
    :param db: The Mongodb database
    :param problem: The dictionary item of the wrong problem
    :param correct: The dictionary item of the correct problem
    :return: None
    """
    # Print questions and get the dictionary mapping printed indices to Mongodb id's
    question_idx_to_id = questions.printlist(db)
    # Display the existing separating questions in the database
    display_separating_questions(db, wrong, correct, question_idx_to_id)
    while True:
        try:
            s = 'Question separating ' + wrong['name'] + ' from ' + correct['name']
            s = s + ' in the DB (0/1)? '
            response = int(raw_input(s))
            break
        except ValueError:
            helper.error_one_zero()
    if response:
        # Separating question is already in our DB; get their hash values
        pos_qhash_list = parse_positive_list_questions(db, wrong, correct, question_idx_to_id)
        neg_qhash_list = parse_negative_list_questions(db, wrong, correct, question_idx_to_id)
    else:
        # Separating question is not in our DB ask for one
        pos_qhash_list = parse_positive_single_question(db, wrong, correct)
        neg_qhash_list = parse_negative_single_question(db, wrong, correct)
    for qhash in pos_qhash_list:
        # For every question set its list correctly and increment its prior and set posterior equal to prior
        set_problem_lists(db, qhash, wrong, correct, True)
        questions.increment(db, qhash, wrong['prior']*correct['prior'])
    for qhash in neg_qhash_list:
        # For every question set its list correctly and increment its prior and set posterior equal to prior
        set_problem_lists(db, qhash, wrong, correct, False)
        questions.increment(db, qhash, wrong['prior']*correct['prior'])


def get_problem_from_list(db):
    """ Ask the user for the correct problem and return its dictionary
    :param db: The Mongodb database
    :return: The dictionary item for the correct problem
    """
    problem_idx_to_id = problems.printlist(db)
    while True:
        try:
            idx = int(raw_input('Enter correct problem number: '))
            correct_id = problem_idx_to_id[idx]
            break
        except ValueError:
            helper.error_number()
        except KeyError:
            helper.error_key()
    correct = db.problems.find_one({'_id': correct_id})
    return correct


def parse_negative_single_question(db, wrong, correct):
    """ Parse a negative single question
    :param db: The Mongodb database
    :param wrong: The dictionary item for the wrong problem
    :param correct: The dictionary item for the correct problem
    :return: List of hash value of the negative question
    """
    # First query the user for a question that is NO for correct problem and YES for wrong one
    q_string = 'Enter a question that is NO for ' + correct['name'] + ' and YES for ' + wrong['name'] + ':\n '
    neg_qname = helper.strip(raw_input(q_string))
    if not neg_qname:
        # User did not enter a new negative question
        return []
    # Get the hash value of the question
    neg_qhashval = helper.gethashval(neg_qname)
    # Check if the question is already in our database
    neg_question = db.questions.find_one({'hash': neg_qhashval})
    if neg_question is None:
        # This is a new negative question, upload to DB
        posproblems = [wrong['hash']]
        negproblems = [correct['hash']]
        prior = 0
        posterior = 0
        loglikelihood = 0.0
        d = {'name': neg_qname, 'hash': neg_qhashval, 'prior': prior,
             'posterior': posterior, 'posproblems': posproblems,
             'negproblems': negproblems, 'loglikelihood': loglikelihood}
        db.questions.insert_one(d)
    return [neg_qhashval]


def parse_positive_single_question(db, wrong, correct):
    """ Parse a positive single question
    :param db: The Mongodb database
    :param problem: The dictionary item for the wrong problem
    :param correct: The dictionary item for the correct problem
    :return: List of hash value of the positive question
    """
    # First query the user for a question that is YES for correct problem and NO for wrong one
    q_string = 'Enter a question that is YES for ' + correct['name'] + ' and NO for ' + wrong['name'] + ':\n '
    pos_qname = helper.strip(raw_input(q_string))
    if not pos_qname:
        # User did not enter a positive question
        return []
    # Get the hash value of the question
    pos_qhashval = helper.gethashval(pos_qname)
    # Check if the question is already in our database
    pos_question = db.questions.find_one({'hash': pos_qhashval})
    if pos_question is None:
        # This is a new positive question, upload to DB with priors and posteriors 0
        posproblems = [correct['hash']]
        negproblems = [wrong['hash']]
        prior = 0
        posterior = 0
        loglikelihood = 0.0
        question = {'name': pos_qname, 'hash': pos_qhashval, 'prior': prior,
             'posterior': posterior, 'posproblems': posproblems,
             'negproblems': negproblems, 'loglikelihood': loglikelihood}
        db.questions.insert_one(question)
    return [pos_qhashval]


def parse_negative_list_questions(db, wrong, correct, question_idx_to_id):
    """ Parse negative questions already in the list
    :param db: The Mongodb database
    :param wrong: The dictionary item for the wrong problem
    :param correct: The dictionary item for the correct problem
    :param question_idx_to_id: The dictionary that maps question numbers to Mongodb ids
    :return: List of hash values of negative questions
    """
    while True:
        try:
            q_string = 'Enter question numbers that are NO for ' \
                       + correct['name'] + ' and YES for ' + wrong['name'] + ':\n '
            neg_list = raw_input(q_string)
            neg_list = map(int, neg_list.strip().split())
            neg_qid_list = [question_idx_to_id[x] for x in neg_list]
            break
        except ValueError:
            helper.error_spaces()
        except KeyError:
            helper.error_key()
    neg_qhash_list = list()
    for item in neg_qid_list:
        hashval = db.questions.find_one({'_id': item})['hash']
        neg_qhash_list.append(hashval)
    return neg_qhash_list


def parse_positive_list_questions(db, wrong, correct, question_idx_to_id):
    """  Get list of hash values of separating questions already in the list
    :param db: The Mongodb database
    :param wrong: The dictionary item for the wrong problem
    :param correct: The dictionary item for the correct problem
    :param question_idx_to_id: The dictionary that maps question numbers to Mongodb ids
    :return: List of hash values of positive questions
    """
    while True:
        try:
            q_string = 'Enter question numbers that are YES for ' \
                       + correct['name'] + ' and NO for ' + wrong['name'] + ':\n '
            pos_list = raw_input(q_string)
            pos_list = map(int, pos_list.strip().split())
            pos_qid_list = [question_idx_to_id[x] for x in pos_list]
            break
        except ValueError:
            helper.error_spaces()
        except KeyError:
            helper.error_key()
    pos_qhash_list = list()
    for item in pos_qid_list:
        hashval = db.questions.find_one({'_id': item})['hash']
        pos_qhash_list.append(hashval)
    return pos_qhash_list


def set_problem_lists(db, qhash, wrong, correct, flag):
    """ Modify the positive and negative lists of the correct and wrong problem
    :param db: The Mongodb database
    :param qhash: The hash value of the question
    :param problem: The dictionary for the wrong problem
    :param correct: The dictionary for the correct problem
    :param flag: Indicates whether it is a positive or a negative separating question
    :return: None, modify the database
    """
    if flag:
        correct_list_name = 'posquestions'
        wrong_list_name = 'negquestions'
    else:
        correct_list_name = 'negquestions'
        wrong_list_name = 'posquestions'
    if '_id' in correct:
        # Add question hash value to appropriate list of the problem
        l = correct[correct_list_name]
        if qhash not in l:
            l.append(qhash)
            db.problems.update({'_id': correct['_id']}, {
                '$set': {correct_list_name: l}
            })
    else:
        # It is a new problem, populate its values
        if flag:
            posquestions = [qhash]
            negquestions = list()
        else:
            posquestions = list()
            negquestions = [qhash]
        correct['posquestions'] = posquestions
        correct['negquestions'] = negquestions
        db.problems.insert_one(correct)
    wrong_list = wrong[wrong_list_name]
    if qhash not in wrong_list:
        wrong_list.append(qhash)
        db.problems.update_one({'_id': wrong['_id']}, {
            '$set': {wrong_list_name: wrong_list}
        })