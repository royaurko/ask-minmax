import gensim
from nltk.data import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import time
import sklearn
import re


class MySentences(object):
    def __init__(self, db):
        """ Constructor for a sentence, used to create an iterator for better memory usage
        :param db: The Mongodb database
        :return: None, create MySentence instance
        """
        self.db = db

    def small_words(self, word_tokens):
        """ Remove all words from word_tokens that are smaller in length than 3, convert to lowercase
        :param word_tokens: Word tokens
        :return: Word tokens with small words removed, converted to lowercase
        """
        return [w.lower() for w in word_tokens if len(w) >= 3]

    def scrunch(self, text):
        """ Remove non-alpha characters from text
        :param text: Text to scrunch
        :return: Scrunched text
        """
        return re.sub('[^a-zA-Z]+', ' ', text)

    def remove_stop(self, word_tokens):
        """ Return new word_token with stop words removed
        :param word_tokens:
        :return:
        """
        return [w for w in word_tokens if w not in stopwords.words('english')]

    def __iter__(self):
        """ Create iterator
        :return: Iterator
        """
        cursor = self.db.papers.find()
        for item in cursor:
            text = item['text']
            # Convert to utf-8
            text = text.encode('utf-8')
            # tokens = self.word_tokenize(text)
            tokens = sent_tokenize(text)
            for sent in tokens:
                words = []
                word_tokens = word_tokenize(sent)
                # Remove non-alpha characters from the words
                for w in word_tokens:
                    scrunched = self.scrunch(w)
                    if scrunched:
                        words.append(scrunched)
                # Remove short words, convert to lower case
                words = self.small_words(words)
                # Remove stop words
                words = self.remove_stop(words)
                # Yield the words to Word2Vec
                if words:
                    yield words


def word2vec_clusters(model):
    """ Run k-means clusters on the Word2vec vectors
    :param model: The word2vec
    :return: None
    """
    start = time.time()
    vectors = model.syn0
    num_clusters = int(raw_input('Number of clusters: '))
    print 'Running clustering with %d clusters' % num_clusters
    clustering_algorithm = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters)
    idx = clustering_algorithm.fit_predict(vectors)
    end = time.time()
    elapsed = end - start
    print("Time for clustering: ", elapsed, "seconds.")
    centroid_map = dict(zip(model.index2word, idx))
    for cluster in range(num_clusters):
        words = [key for key in centroid_map.keys() if centroid_map[key] == cluster]
        if 1 < len(words) < 20:
            print("\nCluster %d" % cluster)
            print(words)


def cluster_tests(db):
    """ Call Word2Vec
    :param db: The Mongodb database
    :return: None, call k-means from w2vClusters(corpus)
    """
    sentences = MySentences(db)
    # For now let the parameters be default
    print("Training word2vec model...")
    model = gensim.models.Word2Vec(sentences, min_count=5)
    # Save the model for future use
    model.save('gensim_model')
    word2vec_clusters(model)
