from __future__ import print_function
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
import time
import sklearn
import re
import os
import logging


class MySentences(object):
    def __init__(self, db, flag):
        """ Constructor for a sentence, used to create an iterator for better memory usage
        :param db: The Mongodb database
        :param flag: If flag is 1 run it on full papers, else on abstracts
        :return: None, create MySentence instance
        """
        self.db = db
        self.flag = flag

    @staticmethod
    def small_words(word_tokens):
        """ Remove all words from word_tokens that are smaller in length than 3, convert to lowercase
        :param word_tokens: Word tokens
        :return: Word tokens with small words removed, converted to lowercase
        """
        return [w.lower() for w in word_tokens if len(w) >= 3]

    @staticmethod
    def scrunch(text):
        """ Remove non-alpha characters from text
        :param text: Text to scrunch
        :return: Scrunched text
        """
        return re.sub('[^a-zA-Z]+', ' ', text)

    @staticmethod
    def remove_stop(word_tokens):
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
        ids = [item['_id'] for item in cursor]
        for id in ids:
            item = self.db.papers.find_one({'_id': id})
            if self.flag:
                text = item['text']
            else:
                text = item['abstract']
            # Convert to utf-8
            text = text.encode('utf-8')
            # tokens = self.word_tokenize(text)
            tokens = sent_tokenize(text)
            sentence_counter = 0
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
                    yield TaggedDocument(words, [sentence_counter])
                    sentence_counter += 1


def word2vec_clusters(model):
    """ Run k-means clusters on the Word2vec vectors
    :param model: The word2vec
    :return: None
    """
    start = time.time()
    vectors = model.syn0
    num_clusters = int(raw_input('Number of clusters: '))
    print("Running clustering with %d clusters" % num_clusters)
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


def build_model(db, flag, cores=1):
    """ Build Word2Vec model
    :param db: The Mongodb database
    :param flag: If flag is 1 then run cluster on full papers, else run on abstracts
    :return: None, call k-means from w2vClusters(corpus)
    """
    sentences = MySentences(db, flag)
    # For now let the parameters be default
    print("Training doc2vec model using %d cores" % cores)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Use phrases
    model = gensim.models.Doc2Vec(sentences, size=500, min_count=10, workers=cores)
    # Save the model for future use
    model_path = 'model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    cursor = db.papers.find(no_cursor_timeout=True)
    model_name = 'model_' + str(cursor.count())
    model.save(model_path + model_name)


def continue_training(db, flag, model_name, cores=1):
    """ Continue training word2vec model
    :param db: Mongodb database
    :param flag: If flag is 1 then train over full texts
    :param model_path: Path to word2vec model
    :param cores: Number of cores to use
    :return: None
    """
    sentences = MySentences(db, flag)
    print("Training doc2vec model using %d cores" % cores)
    # Use phrases
    bigram_transformer = gensim.models.Phrases(sentences)
    # Load model from model_path
    model = gensim.models.Doc2Vec.load(model_name)
    # Continue training model
    model.train(sentences)
    # Save the model for future use
    model_path = 'model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    cursor = db.papers.find()
    model_name = 'model_' + str(cursor.count())
    model.save(model_path + model_name)