import helper
import problems
import sepquestions


def train(db):
    ''' Query the number of times to train, then call training()
    :param db: The Mongodb database
    :return: None, call training
    '''
    while True:
        try:
            n = int(raw_input('Number of times to train? '))
            break
        except ValueError:
            helper.errornumber()
    training(n, db)


def training(n, db):
   ''' Sample a problem according to its prior and ask whether it was a correct guess or not
   :param n: Number of times to train
   :param db: The Mongodb database
   :return: None, modify db in place
   '''
   for i in xrange(n):
        problem = problems.sampleproblem(db, 'prior')
        print problem['name']
        while True:
            try:
                response = int(raw_input('Is this the correct problem (0/1)? '))
                break
            except ValueError:
                helper.erroronezero()
        if response:
            # Increase the count and update the priors and posteriors of this problem
            problems.increment(db, problem['hash'])
            continue
        else:
            # It was not the right guess, ask for a separating question
            sepquestions.separatingquestion(db, problem)
