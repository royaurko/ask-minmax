# Contains all functions pertaining to problems

import helper
import random


def printproblemlist(db):
    cursor = db.problems.find()
    eq = '=' * 40
    print eq
    print 'List of problems'
    i = 1
    order = dict()
    for item in cursor:
        print str(i) + '. ' + item['name']
        order[i] = item['_id']
        i += 1
    print eq
    return order


def sampleproblem(db, p):
    '''Sample problems proportional to its prior value'''
    cursor = db.problems.find()
    count = cursor.count()
    if count < 1:
        '''Trying to sample from empty collection'''
        print 'Empty problem set!'
        queryproblems(db)
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


def queryproblems(db):
    '''Query problems together with priors to add to database'''
    flag = 1
    while flag:
        pname = raw_input('Problem name: ')
        pname = helper.strip(pname)
        tokens = helper.gettokens(pname)
        item = db.problems.find_one({'tokens': tokens})
        if item is None:
            prior = 1.0
            posquestions = list()
            negquestions = list()
            posterior = 1.0
            d = {'name': pname, 'tokens': tokens, 'prior': prior,
                 'posterior': posterior, 'posquestions': posquestions,
                 'negquestions': negquestions}
            db.problems.insert(d)
        while True:
            try:
                flag = int(raw_input('Continue (0/1)?'))
                break
            except ValueError:
                helper.erroronezero()


def maxproblemposterior(db):
    '''Return the value of the maximum posterior'''
    cursor = db.problems.find()
    m = 0.0
    total = 0.0
    for item in cursor:
        total += item['posterior']
        if item['posterior'] > m:
            m = item['posterior']
    m /= total
    return m
