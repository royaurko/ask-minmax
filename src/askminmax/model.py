from __future__ import print_function
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
import gensim
from gensim import utils
import time
import sklearn
import os
import logging
from sklearn.linear_model import LogisticRegression
import multiprocessing
import random
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout, Activation
num_cpu = multiprocessing.cpu_count()


class MySentences(object):
    def __init__(self, data_set):
        """ Constructor for a sentence, used to create an iterator for better memory usage
        :param data_set: The data set containing the problems as subfolders
        :return: None, create MySentence instance
        """
        self.data_set = data_set
        self.sentences = []

    def to_array(self):
        self.sentences = []
        keywords = os.listdir(self.data_set)
        for keyword in keywords:
            count = 0
            for index, abstract in enumerate(os.listdir(self.data_set + '/' + keyword)):
                with utils.smart_open(self.data_set + '/' + keyword + '/' + abstract) as fin:
                    for item_no, line in enumerate(fin):
                        label = keyword.replace(' ', '_')
                        self.sentences.append(TaggedDocument(utils.to_unicode(line).split(),
                                                             label + '_%s' % count))
                        count += 1
        return self.sentences

    def sentences_perm(self):
        return np.random.permutation(self.sentences)

    def __iter__(self):
        """ Create iterator
        :return: Iterator
        """
        keywords = os.listdir(self.data_set)
        for keyword in keywords:
            count = 0
            for abstract in os.listdir(self.data_set + '/' + keyword):
                with utils.smart_open(self.data_set + '/' + keyword + '/' + abstract) as fin:
                    for line in fin:
                        label = keyword.replace(' ', '_')
                        yield TaggedDocument(utils.to_unicode(line).split(), label + '_%s' % count)
                        count += 1


def get_dense_model(input_dim, output_dim):
    """ Build a simple MLP to classify Doc2Vec vectors to problems
    :param input_dim:
    :param output_dim:
    :return:
    """
    nb_dim = 1024
    model = Sequential()
    # First dense layer
    model.add(Dense(input_dim, nb_dim))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    # Second dense layer
    model.add(Dense(nb_dim, nb_dim))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    # Third dense layer
    model.add(Dense(nb_dim, output_dim))
    model.add(Activation('softmax'))
    # Add optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def fit_mlp(db, dimension, keywords, mlp_model, doc2vec_model, batch_size=32, nb_epoch=10):
    """ MLP classifier
    :param db: MongoDB database
    :param dimension: The dimension of the word vectors
    :param keywords: The problems/keywords
    :param mlp_model: MLP model
    :param doc2vec_model: Doc2Vec model
    :param batch_size: Batch size of training
    :param nb_epoch: Number of epochs to train for
    :return: Trained MLP model
    """
    key_count, total_count = 0, 0
    num_documents = db.papers.find().count()
    train_arrays = np.zeros((num_documents, dimension))
    train_labels = np.zeros(num_documents)
    for keyword in keywords:
        cursor = db.papers.find({'keyword': keyword}, no_cursor_timeout=True)
        for i in xrange(cursor.count()):
            tag = keyword + '_' + total_count
            train_arrays[total_count] = doc2vec_model[tag]
            train_labels[total_count] = key_count
            total_count += 1
        key_count += 1
        cursor.close()
    check_pointer = ModelCheckpoint(filepath='model/mlp_weights', verbose=1, save_best_only=True)
    history = mlp_model.fit(train_arrays, train_labels, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
                            show_accuracy=True, validation_split=0.1, callbacks=[check_pointer])
    return history


def predict(db, dimension, keywords, doc2vec_model, text):
    """
    :param db:
    :param dimension:
    :param keywords:
    :param model:
    :param text:
    :return:
    """
    mlp_model = get_dense_model(dimension, len(keywords))


def get_classifier(data_set, dimension, model):
    """ Logistic regression classifier
    :param data_set: Path to data set folder
    :param dimension: Dimension of the word vectors
    :param model: Doc2Vec model
    :return:
    """
    key_count, total_count = 0, 0
    keywords = os.listdir(data_set)
    num_documents = 0
    for keyword in keywords:
        num_documents += len(os.listdir(data_set + '/' + keyword))
    train_arrays = np.zeros((num_documents, dimension))
    train_labels = np.zeros(num_documents)
    for keyword in keywords:
        for abstract in os.listdir(data_set + '/' + keyword):
            f = open(data_set + '/' + keyword + '/' + abstract, 'rb')
            text = f.read()
            vector = model.infer_vector(text)
            train_arrays[total_count] = vector
            train_labels[total_count] = key_count
            total_count += 1
        key_count += 1
    classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                    intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
    classifier.fit(train_arrays, train_labels)
    return classifier


def classify(data_set, model, classifier, text):
    """ Classify using logistic regression
    :param data_set: Path to dataset
    :param model: Doc2Vec model
    :param classifier: Logistic Regression classifier
    :param text: Text to classify
    :return:
    """
    keywords = os.listdir(data_set)
    idx_to_keywords = dict([(i, v) for i, v in enumerate(keywords)])
    vector = model.infer_vector(text)
    probability_vector = classifier.predict_proba(vector).flatten
    return dict([(idx_to_keywords[i], v) for i, v in enumerate(probability_vector)])


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


def build_model(data_set, cores=num_cpu, num_epochs=10):
    """ Build Word2Vec model
    :param data_set: The dataset folder containing the problems
    :param cores: Number of cores to use
    :param num_epochs: Number of epochs to train for
    :return: None, call k-means from w2vClusters(corpus)
    """
    sentences = MySentences(data_set)
    print("Training doc2vec model using %d cores for %d epochs" % (cores, num_epochs))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Doc2Vec(sentences, size=400, min_count=1, window=10, alpha=0.025, min_alpha=0.025,
                                  sample=1e-4, workers=cores)
    model.build_vocab(sentences.to_array())
    sentences_list = sentences.to_array()
    idx = range(len(sentences_list))
    for i in xrange(num_epochs):
        random.shuffle(idx)
        perm_sentences = [sentences_list[i] for i in idx]
        model.train(perm_sentences)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    # Save model
    model_path = 'model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    time_str = time.strftime("_%Y-%m-%d_%H-%M-%S")
    model_name = 'model_' + time_str + '.d2v'
    model.save(model_path + model_name)


def continue_training(db, flag, model_name, cores=num_cpu, num_epochs=10):
    """ Continue training word2vec model
    :param db: Mongodb database
    :param flag: If flag is 1 then train over full texts
    :param model_name: Path to word2vec model
    :param cores: Number of cores to use
    :param num_epochs: Number of epochs to train for
    :return: None
    """
    # Get keywords
    cursor = db.papers.find(no_cursor_timeout=True)
    keywords = [item['keyword'] for item in cursor]
    sentences = MySentences(db, flag, keywords)
    print("Training doc2vec model using %d cores for %d epochs" % (cores, num_epochs))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Load model from model_path
    model = gensim.models.Doc2Vec.load(model_name)
    sentences_list = sentences.to_array()
    idx = range(len(sentences_list))
    for i in xrange(num_epochs):
        random.shuffle(idx)
        perm_sentences = [sentences_list[i] for i in idx]
        model.train(perm_sentences)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    # Save the model for future use
    model_path = 'model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    time_str = time.strftime("_%Y-%m-%d_%H-%M-%S")
    model_name = 'model_' + str(cursor.count()) + time_str
    model.save(model_path + model_name)
    cursor.close()