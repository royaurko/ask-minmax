# Contains functions responsible for training

import helper
import problems
import sepquestions


def train(db):
    '''Query number of times to train, then call training'''
    while True:
        try:
            n = int(raw_input('Number of times to train?'))
            break
        except ValueError:
            helper.errornumber()
    training(n, db)


def training(n, db):
    '''n is the number of trials, q is the set of questions known till now'''
    for i in range(n):
        problem = problems.sampleproblem(db, 'prior')
        print problem['name']
        while True:
            try:
                flag = int(raw_input('Is this the correct problem (0/1)? '))
                break
            except ValueError:
                helper.erroronezero()
        if flag:
            # Increase the count and update the priors and posteriors of this problem
            problems.increasecount(db, problem)
            problems.updatepriors(db, problem)
            problems.updateposteriors(db, problem)
            continue
        else:
            sepquestions.separatingquestion(db, problem)
