from . import helper
from . import problems
from . import sepquestions


def train(db):
    """ Query the number of times to train, then call training()
    :param db: The Mongodb database
    :return: None, call training
    """
    while True:
        try:
            n = eval(input('Number of times to train? '))
            break
        except ValueError:
            helper.error_number()
    training(n, db)


def training(n, db):
    """ Sample a problem according to its prior and ask whether it was a correct guess or not
    :param n: Number of times to train
    :param db: The Mongodb database
    :return: None, modify db in place
    """
    for i in range(n):
        problem = problems.sample(db, 'prior')
        print((problem['name']))
        while True:
            try:
                response = eval(input('Is this the correct problem (0/1)? '))
                break
            except ValueError:
                helper.error_one_zero()
        if response:
            # Increase the count and update the priors and posteriors of this problem
            problems.increment(db, problem['hash'])
            # Normalize the priors and posteriors of the problems
            problems.normalize(db, 'prior')
            problems.normalize(db, 'posterior')
            continue
        else:
            # It was not the right guess: get the correct problem
            correct, correct_hash = sepquestions.get_correct_problem(db)
            # Ask for a separating question
            sepquestions.ask_separating_question(db, problem, correct, correct_hash)
            # Increment the prior for the correct problem and set its posterior equal to prior
            problems.increment(db, correct_hash)
            # Normalize priors and posteriors of problems
            problems.normalize(db, 'prior')
            problems.normalize(db, 'posterior')