# Contains functions pertaining to questions

import random


def maxquestionposterior(db):
    '''Return the value of the maximum posterior'''
    cursor = db.questions.find()
    m = 0.0
    total = 0.0
    for item in cursor:
        total += item['posterior']
        if item['posterior'] > m:
            m = item['posterior']
    if total:
        m /= total
    return m


def printposteriors(db):
    '''Print list of questions with their posteriors'''
    cursor = db.questions.find()
    for item in cursor:
        print item['name'] + ': ' + str(item['posterior'])


def printquestionlist(db):
    cursor = db.questions.find()
    eq = '=' * 40
    print eq
    print 'List of questions'
    i = 1
    order = dict()
    for item in cursor:
        print str(i) + '. ' + item['name']
        order[i] = item['_id']
        i += 1
    print eq
    return order


def samplequestion(db, p):
    '''Sample questions proportional to its p value'''
    cursor = db.questions.find()
    count = cursor.count()
    zerocount = db.questions.find({p: 0.0}).count()
    if count < 1 or count == zerocount:
        '''Trying to sample from empty question set'''
        print 'No questions with non-zero posterior!'
        return
    weight = 0
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
