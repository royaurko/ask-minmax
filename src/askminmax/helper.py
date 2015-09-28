from __future__ import print_function
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
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


def small_words(word_tokens):
    """ Remove all words from word_tokens that are smaller in length than 3, convert to lowercase
    :param word_tokens: Word tokens
    :return: Word tokens with small words removed, converted to lowercase
    """
    return [w.lower() for w in word_tokens if len(w) >= 3]


def scrunch(text):
    """ Remove non-alpha characters from text
    :param text: Text to scrunch
    :return: Scrunched text
    """
    return re.sub('[^a-zA-Z]+', ' ', text)


def remove_stop(word_tokens):
    """ Return new word_token with stop words removed
    :param word_tokens:
    :return:
    """
    return [w for w in word_tokens if w not in stopwords.words('english')]


def clean_text(db, flag, mongo_id):
        """ Clean up text given its mongodb id
        :param mongo_id: Mongodb id of item
        :return: None, clean up the db in place
        """
        item = db.papers.find_one({'_id': mongo_id})
        print('Cleaning up entry ', mongo_id)
        if flag:
            text = item['text']
        else:
            text = item['abstract']
        text = scrunch(text)
        tokens = sent_tokenize(text)
        sentences = []
        for sent in tokens:
            words = word_tokenize(sent)
            # Remove short words, convert to lower case
            words = small_words(words)
            # Remove stop words
            words = remove_stop(words)
            words_str = ' '.join(words)
            sentences.append(words_str)
        if flag:
            item['text'] = '.'.join(sentences)
        else:
            item['abstract'] = '.'.join(sentences)
        db.papers.update({'_id': item['_id']}, {"$set": item}, upsert=False)
