import hashlib
import re


def errorspaces():
    ''' Error when user does not enter space separated question numbers
    :return: None
    '''
    print 'Please separate question numbers by spaces!'


def erroronezero():
    ''' Error when user enters something other than 0 or 1
    :return: None
    '''
    print 'Please enter 0 or 1!'


def errornumber():
    ''' Error when user does not enter a number
    :return:None
    '''
    print 'Please enter a valid number!'


def errorkey():
    ''' Error if user does not enter correct number
    :return: None
    '''
    print 'Please enter a correct number from the list!'

def strip(s):
    ''' Strip spaces from left and right
    :param s: string to strip
    :return: stripped string
    '''
    s = s.lstrip()
    s = s.rstrip()
    return s


def gethashval(s):
    '''
    :param s: A string s corresponding to the name of a problem or a question
    :return: A SHA 256 hash of the name of the problem after some preprocessing
    '''
    regex = re.compile('[^a-zA-Z]')
    s = regex.sub('', s)
    s = s.lower()
    return hashlib.md5(s).hexdigest()


def mass(db, table, hashval, property):
    ''' Return mass of property of query is non-zero and 0 o.w.
    :param db: The Mongodb database
    :param table: Type of db table, either problems or questions
    :param tokens:
    :param property: Property type whose mass to return
    :return: Mass of the property
    '''
    if table == 'problems':
        problem = db.problems.find_one({'hash': hashval})
        return problem[property]
    elif table == 'questions':
        question = db.problems.find_one({'hash': hashval})
        return question[property]