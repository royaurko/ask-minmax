from __future__ import print_function
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
import gensim
import time
import sklearn
import re
import os
import logging
from sklearn.linear_model import LogisticRegression


class MySentences(object):
    def __init__(self, db, flag, keywords):
        """ Constructor for a sentence, used to create an iterator for better memory usage
        :param db: The Mongodb database
        :param flag: If flag is 1 run it on full papers, else on abstracts
        :return: None, create MySentence instance
        """
        self.db = db
        self.flag = flag
        self.keywords = keywords
        self.sentences = []

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

    def clean_up_db(self):
        """ Clean up the database by removing stop words etc
        :return: None
        """
        # Remove non-alpha characters from the words
        cursor = self.db.papers.find()
        count = 1
        for item in cursor:
            print('Cleaning up entry ', count)
            count += 1
            if self.flag:
                text = item['text']
            else:
                text = item['abstract']
            text = self.scrunch(text)
            tokens = sent_tokenize(text)
            sentences = []
            for sent in tokens:
                words = word_tokenize(sent)
                # Remove short words, convert to lower case
                words = self.small_words(words)
                # Remove stop words
                words = self.remove_stop(words)
                words_str = ' '.join(words)
                sentences.append(words_str)
            if self.flag:
                item['text'] = '.'.join(sentences)
            else:
                item['abstract'] = '.'.join(sentences)
            self.db.papers.update({'_id': item['_id']}, item)

    def to_array(self):
        self.sentences = []
        for keyword in self.keywords:
                cursor = self.db.papers.find({'keyword': keyword})
                for item in cursor:
                    if self.flag:
                        text = item['text']
                    else:
                        text = item['abstract']
                    # Convert to utf-8
                    try:
                        text = text.decode('utf8')
                    except UnicodeDecodeError:
                        print('Error decoding ' + text)
                        print(type(text))
                    # Tokenize sentences
                    tokens = sent_tokenize(text)
                    sentence_counter = 0
                    for sent in tokens:
                        word_tokens = word_tokenize(sent)
                        # Append to sentences
                        if word_tokens:
                            self.sentences.append(TaggedDocument(word_tokens, [keyword + '_' + str(sentence_counter)]))
                            sentence_counter += 1
        return self.sentences

    def sentences_perm(self):
        return np.random.permutation(self.sentences)

    def __iter__(self):
        """ Create iterator
        :return: Iterator
        """
        for keyword in self.keywords:
            cursor = self.db.papers.find({'keyword': keyword})
            ids = [item['_id'] for item in cursor]
            for id in ids:
                item = self.db.papers.find_one({'_id': id})
                if self.flag:
                    text = item['text']
                else:
                    text = item['abstract']
                # Convert to utf-8
                try:
                    text = text.decode('utf8')
                except UnicodeDecodeError:
                    print('Error decoding ' + text)
                    print(type(text))
                # tokens = self.word_tokenize(text)
                tokens = sent_tokenize(text)
                sentence_counter = 0
                for sent in tokens:
                    word_tokens = word_tokenize(sent)
                    # Yield the words to Word2Vec
                    if word_tokens:
                        yield TaggedDocument(word_tokens, [keyword + '_' + str(sentence_counter)])
                        sentence_counter += 1


def get_classifier(db, dimension, keywords, model):
    key_count, total_count = 0, 0
    num_documents = db.papers.find().count()
    train_arrays = np.zeros((num_documents, dimension))
    train_labels = np.zeros(num_documents)
    for keyword in keywords:
        cursor = db.papers.find({'keyword': keyword})
        for i in xrange(cursor.count()):
            tag = keyword + '_' + total_count
            train_arrays[total_count] = model[tag]
            train_labels[total_count] = key_count
            total_count += 1
        key_count += 1
    classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', multi_class='ovr', random_state=None, tol=0.0001)
    classifier.fit(train_arrays, train_labels)
    return classifier


def classify(db, dimension, keywords, model, text):
    classifier = get_classifier(db, dimension, keywords, model)
    idx_to_keywords = dict([(i, v) for i, v in enumerate(keywords)])
    vector = model.infer_vector(text)
    label = classifier.predict(vector)
    return idx_to_keywords[label]


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


def build_model(db, clean_flag, flag, cores=1, num_epochs=10):
    """ Build Word2Vec model
    :param db: The Mongodb database
    :param flag: If flag is 1 then run cluster on full papers, else run on abstracts
    :return: None, call k-means from w2vClusters(corpus)
    """
    # Get keywords
    cursor = db.papers.find()
    keywords = [item['keyword'] for item in cursor]
    sentences = MySentences(db, flag, keywords)
    # Clean up the db
    if clean_flag:
        print('Cleaning up the db')
        sentences.clean_up_db()
    # For now let the parameters be default
    print("Training doc2vec model using %d cores" % cores)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Doc2Vec(sentences, size=400, min_count=1, window=10, alpha=0.025, min_alpha=0.025,
                                  sample=1e-4, workers=cores)
    model.build_vocab(sentences.to_array())
    for i in xrange(num_epochs):
        model.train(sentences.sentences_perm())
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    # Save model
    model_path = 'model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    cursor = db.papers.find()
    model_name = 'model_' + str(cursor.count()) + '_' + str(num_epochs)
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
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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