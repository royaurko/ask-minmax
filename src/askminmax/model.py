from gensim.models.doc2vec import TaggedDocument
import numpy as np
import gensim
from gensim import utils
import time
import sklearn
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import multiprocessing as mp
import pickle
import argparse
import random
num_cpu = mp.cpu_count()


class MySentences:
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


def get_vector(doc2vec_model, data_set, keyword, abstract, d):
    """ Get the vector for the abstract
    :param doc2vec_model: Doc2Vec model
    :param data_set: Data set containing abstracts
    :param keyword: Keyword
    :param abstract: Abstract
    :param d: Dictionary mapping keyword to label
    :return:
    """
    f = open(data_set + '/' + keyword + '/' + abstract, encoding='utf-8')
    text = f.read()
    f.close()
    vector = doc2vec_model.infer_vector(text)
    label = d[keyword]
    return vector, label


def train_random_forest_classifier(data_set, doc2vec_model_path):
    """ Random forest classifier
    :param data_set: Path to data set folder
    :param doc2vec_model_path: Path to Doc2Vec model
    :return: None, save classifier to pickle file
    """
    print('Building a random forest classifier...')
    doc2vec_model = gensim.models.Doc2Vec.load(doc2vec_model_path)
    keywords = os.listdir(data_set)
    d = {}
    for i, keyword in enumerate(keywords):
        d[keyword] = i
    data = []
    for keyword in keywords:
        for abstract in os.listdir(data_set + '/' + keyword):
            labelled_vector = get_vector(doc2vec_model, data_set, keyword, abstract, d)
            data.append(labelled_vector)
    # Split train and test
    random.shuffle(data)
    n = len(data)
    m = int(0.9*n)
    train_data = data[: m]
    test_data = data[m: n]
    train_arrays = np.array([x[0] for x in train_data])
    train_labels = np.array([x[1] for x in train_data])
    test_arrays = np.array([x[0] for x in test_data])
    test_labels = np.array([x[1] for x in test_data])
    classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    classifier.fit(train_arrays, train_labels)
    print('Classifier score on train data = ', classifier.score(train_arrays, train_labels))
    print('Classifier score on test_data = ', classifier.score(test_arrays, test_labels))
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_name = 'model_' + time_str + '.rf'
    f = open(model_path + model_name, 'wb')
    pickle.dump(classifier, f)
    f.close()


def train_logistic_regression_classifier(data_set, doc2vec_model_path):
    """ Logistic regression classifier
    :param data_set: Path to data set folder
    :param doc2vec_model_path: Path to Doc2Vec model
    :return: None, save classifier to pickle file
    """
    print('Building a logistic regression classifier...')
    doc2vec_model = gensim.models.Doc2Vec.load(doc2vec_model_path)
    keywords = os.listdir(data_set)
    d = {}
    for i, keyword in enumerate(keywords):
        d[keyword] = i
    # Collect vectors in parallel
    data = []
    for keyword in keywords:
        for abstract in os.listdir(data_set + '/' + keyword):
            vector = get_vector(doc2vec_model, data_set, keyword, abstract, d)
            data.append(vector)
    # Split train and test
    random.shuffle(data)
    n = len(data)
    m = int(0.9*n)
    train_data = data[: m]
    test_data = data[m: n]
    # Do a cross validation
    train_arrays = np.array([x[0] for x in train_data])
    train_labels = np.array([x[1] for x in train_data])
    test_arrays = np.array([x[0] for x in test_data])
    test_labels = np.array([x[1] for x in test_data])
    classifier = LogisticRegression(C=1.0, max_iter=1000, penalty='l2',
                                      solver='newton-cg')
    classifier.fit(train_arrays, train_labels)
    print('Classifier score on train_data = ', classifier.score(train_arrays, train_labels))
    print('Classifier score on test_data = ', classifier.score(test_arrays, test_labels))
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_name = 'model_' + time_str + '.log'
    f = open(model_path + model_name, 'wb')
    pickle.dump(classifier, f)
    f.close()


def classify(data_set, doc2vec_model_path, classifier_path, text):
    """ Classify using logistic regression
    :param data_set: Path to dataset
    :param doc2vec_model_path: Path to Doc2Vec model
    :param classifier_path: Path to Random forest classifier
    :param text: Text to classify
    :return:
    """
    logistic_file = open(classifier_path, 'rb')
    classifier = pickle.load(logistic_file, encoding='latin1')
    model = gensim.models.Doc2Vec.load(doc2vec_model_path)
    keywords = os.listdir(data_set)
    idx_to_keywords = dict([(i, v) for i, v in enumerate(keywords)])
    vector = model.infer_vector(text).reshape(1, -1)
    probability_vector = classifier.predict_proba(vector).flatten()
    logistic_file.close()
    return dict([(idx_to_keywords[i], v) for i, v in enumerate(probability_vector)])


def word2vec_clusters(model):
    """ Run k-means clusters on the Word2vec vectors
    :param model: The word2vec
    :return: None
    """
    start = time.time()
    vectors = model.syn0
    num_clusters = eval(input('Number of clusters: '))
    print("Running clustering with {0} clusters".format(num_clusters))
    clustering_algorithm = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters)
    idx = clustering_algorithm.fit_predict(vectors)
    end = time.time()
    elapsed = end - start
    print(("Time for clustering: ", elapsed, "seconds."))
    centroid_map = dict(list(zip(model.index2word, idx)))
    for cluster in range(num_clusters):
        words = [key for key in list(centroid_map.keys()) if centroid_map[key] == cluster]
        if 1 < len(words) < 20:
            print("\nCluster {0}".format(cluster))
            print(words)


def build_model(data_set, cores=num_cpu, num_epochs=10):
    """ Build Word2Vec model
    :param data_set: The dataset folder containing the problems
    :param cores: Number of cores to use
    :param num_epochs: Number of epochs to train for
    :return: None, call k-means from w2vClusters(corpus)
    """
    sentences = MySentences(data_set)
    print("Training doc2vec model using {0} cores for {1} epochs".format(cores, num_epochs)))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Doc2Vec(sentences, size=400, min_count=10, window=10, alpha=0.025, min_alpha=0.025,
                                  sample=1e-4, workers=cores)
    model.build_vocab(sentences.to_array())
    sentences_list = sentences.to_array()
    idx = list(range(len(sentences_list)))
    for i in range(num_epochs):
        random.shuffle(idx)
        perm_sentences = [sentences_list[i] for i in idx]
        model.train(perm_sentences)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    # Save model
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
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
    print("Training doc2vec model using {0} cores for {0} epochs".format(cores, num_epochs))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Load model from model_path
    model = gensim.models.Doc2Vec.load(model_name)
    sentences_list = sentences.to_array()
    idx = list(range(len(sentences_list)))
    for i in range(num_epochs):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="'doc2vec' for training doc2vec vectors, 'logit' for building logit model, 'rf' for RF model")
    parser.add_argument('-n', '--num_cores', help='Number of cores to use with -w')
    parser.add_argument('-e', '--num_epochs', help='Number of epochs to train with -w')
    parser.add_argument('-d', '--data_path', help='Path to abstracts')
    parser.add_argument('-m', '--model_path', help='Path to Doc2Vec model')
    args = parser.parse_args()
    if args.mode == 'doc2vec':
        try:
            build_model(args.data_path, args.num_cores, args.num_epochs)
        except:
            print('Unexpected error')
    if args.mode == 'logit':
        try:
            train_logistic_regression_classifier(args.data_path, args.model_path)
        except Exception as e:
            print('Unexpected error: ', e)
    if args.mode == 'rf':
        try:
            train_random_forest_classifier(args.data_path, args.model_path)
        except Exception as e:
            print('Unexpected error: ', e)
