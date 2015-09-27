from __future__ import print_function
import hashlib
import re


def error_spaces():
    """ Error when user does not enter space separated question numbers
    :return: None
    """
    print('Please separate question numbers by spaces!')


def error_one_zero():
    """ Error when user enters something other than 0 or 1
    :return: None
    """
    print('Please enter 0 or 1!')


def error_number():
    """ Error when user does not enter a number
    :return:None
    """
    print('Please enter a valid number!')


def error_key():
    """ Error if user does not enter correct number
    :return: None
    """
    print('Please enter a correct number from the list!')


def strip(s):
    """ Strip spaces from left and right
    :param s: string to strip
    :return: stripped string
    """
    s = s.lstrip()
    s = s.rstrip()
    return s


def get_hash(s):
    """
    :param s: A string s corresponding to the name of a problem or a question
    :return: A SHA 256 hash of the name of the problem after some preprocessing
    """
    regex = re.compile('[^a-zA-Z]')
    s = regex.sub('', s)
    s = s.lower()
    return hashlib.md5(s).hexdigest()


def mass(db, table, hash_value, property_name):
    ''' Return mass of property of query is non-zero and 0 o.w.
    :param db: The Mongodb database
    :param table: Type of db table, either problems or questions
    :param hash_value: Hash value of the problem or question
    :param property_name: Property type whose mass to return
    :return: Mass of the property
    '''
    if table == 'problems':
        problem = db.problems.find_one({'hash': hash_value})
        return problem[property_name]
    elif table == 'questions':
        question = db.problems.find_one({'hash': hash_value})
        return question[property_name]


def get_initials(s):
    """ Get the initials of the string s
    :param s: String to get initials of
    :return: The initials
    """
    initials = ''
    s_list = s.strip().split()
    for i in xrange(len(s_list)):
        initials += s_list[i][0]
    return initials.upper()