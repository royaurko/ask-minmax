import gensim
from nltk.data import load
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
import time
import sklearn
import re


class MySentences(object):
    def __init__(self, db):
        ''' Constructor for a sentence, used to create an iterator for better memory usage
        :param db: The Mongodb database
        :return: None, create MySentence instance
        '''
        self.db = db


    def sent_tokenize(self, text):
        ''' Return sentence tokenized copy of text
        :param text: The text to tokenize
        :param language: default is english
        :return: Tokens
        '''
        tokenizer = load('tokenizers/punkt/{0}.pickle'.format('english'))
        return tokenizer.tokenize(text)


    def word_tokenize(self, text):
        ''' Given sentence as a text return a token of words
        :param text: A sentence
        :return: Token of words
        '''
        return [token for sent in self.sent_tokenize(text)
            for token in TreebankWordTokenizer().tokenize(sent)]


    def smallwords(self, word_tokens):
        ''' Remove all words from word_tokens that are smaller in length than 3, convert to lowercase
        :param word_tokens: Word tokens
        :return: Word tokens with small words removed, converted to lowercase
        '''
        return [w.lower() for w in word_tokens if len(w) >= 3]


    def scrunch(self, text):
        ''' Remove non-alpha characters from text
        :param text: Text to scrunch
        :return: Scrunched text
        '''
        return re.sub('[^a-zA-Z]+', ' ', text)


    def removestop(self, word_tokens):
        ''' Return new word_token with stop words removed
        :param word_tokens:
        :return:
        '''
        return [w for w in word_tokens if w not in stopwords.words('english')]


    def __iter__(self):
        ''' Create iterator
        :return:
        '''
        cursor = self.db.papers.find()
        for item in cursor:
            text = item['text']
            # Convert to utf-8
            text = text.encode('utf-8')
            tokens = self.word_tokenize(text)
            cleaned_tokens = []
            for sent in tokens:
                words = []
                # Remove non-alpha characters from the words
                for w in sent:
                    words.append(self.scrunch(w))
                # Remove short words, convert to lower case
                words = self.smallwords(words)
                # Remove stop words
                words = self.removestop(words)
                # Yield the words to Word2Vec
                yield words



def w2vClusters(model):
    ''' Run k-means clusters on the Word2vec vectors
    :param model: The word2vec
    :return: None
    '''
    start = time.time()
    vectors = model.syn0
    numC = int(raw_input('Number of clusters: '))
    print 'Running clustering with %d clusters' % numC
    clusterAlgo = sklearn.cluster.MiniBatchKMeans(n_clusters=numC)
    idx = clusterAlgo.fit_predict(vectors)
    end = time.time()
    elapsed = end - start
    print("Time for clustering: ", elapsed, "seconds.")
    centroidMap = dict(zip(model.index2word, idx))
    for cluster in range(numC):
        words = [key for key in centroidMap.keys() if centroidMap[key] == cluster]
        if len(words) > 1 and len(words) < 20:
            print("\nCluster %d" % cluster)
            print(words)


def clusterTests(db):
    ''' Call Word2Vec
    :param db: The Mongodb database
    :return: None, call k-means from w2vClusters(corpus)
    '''
    sentences = MySentences(db)
    # For now le the parameters be default
    print("Training word2vec model...")
    model = gensim.models.Word2Vec(sentences)
    w2vClusters(model)