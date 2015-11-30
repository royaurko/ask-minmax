from . import helper
from . import database
from . import problems
from . import questions
from . import sepquestions
from . import training
from . import arxiv
from . import natural_break
import numpy as np
from . import model
from functools import reduce
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from scipy.stats import entropy
from jenks import jenks
import gensim
import os


class Expert:
    def __init__(self):
        """ Constructor for Expert class
        :return: None
        """
        while True:
            try:
                build_question = 'Build new database (0/1)? '
                response = eval(input(build_question))
                break
            except (SyntaxError, NameError, ValueError):
                helper.error_one_zero()
        client, db = database.initialize_db()
        if response:
            problems.query(db)
            training.train(db)
        else:
            while True:
                try:
                    recover_question = 'Recover a database from bson file? '
                    flag = eval(input(recover_question))
                    break
                except (SyntaxError, NameError, ValueError):
                    helper.error_one_zero()
            if flag:
                db = database.recover_db(client)
        # Set the expert instance database to db
        self.db = db

    def train(self):
        """ Call the train subroutine from training to learn separating questions
        :return: None, modify db in place
        """
        training.train(self.db)

    def delete(self):
        """ Allows user to delete a problem or a question from the database
        :return: None, modify database in place
        """
        db = self.db
        problem_idx_to_id = problems.print_list(db)
        try:
            problems_list = eval(input('Enter indices of problems to delete separated by comma: '))
        except (NameError, ValueError, SyntaxError):
            helper.error_comma()
        if type(problems_list) is int:
            problems.delete(db, problem_idx_to_id[problems_list])
        else:
            for problem in problems_list:
                problems.delete(db, problem_idx_to_id[problem])
        question_idx_to_id = questions.print_list(db)
        try:
            questions_list = eval(input('Enter indices of questions to delete separated by comma: '))
        except (NameError, ValueError, SyntaxError):
            helper.error_comma()
        if type(questions_list) is int:
            questions.delete(db, question_idx_to_id[questions_list])
        else:
            for question in questions_list:
                questions.delete(db, question_idx_to_id[question])
        print('Modified database:')
        self.print_table()

    def print_table(self):
        """ Print the current list of problems and questions with their priors and posteriors
        :return: None
        """
        db = self.db
        problems.print_list(db)
        questions.print_list(db)

    def run(self, data_set, doc2vec_model_path, classifier_path):
        """ Control the main program flow
        :param data_set: Path to data set
        :param doc2vec_model_path: Path to doc2vec model
        :param classifier_path: Path to classifier
        :return: None, modify db in place
        """
        try:
            while True:
                # Ask the user for a short summary of the problem
                summary = input('Please describe your problem in a few words: ')
                # Adjust problem priors
                self.adjust_problem_priors_from_summary(data_set, doc2vec_model_path, classifier_path, summary)
                # Adjust question priors
                # self.adjust_question_priors_from_summary(summary, doc2vec_model_path)
                # Reset the posteriors equal to prior before starting a prediction loop
                self.reset_posteriors()
                # Print the table
                self.print_table()
                # Call control_prediction
                self.control_prediction()
                print('Press [Ctrl] + c to exit')
        except KeyboardInterrupt:
            self.query_backup()

    def reset_posteriors(self):
        """ Reset posteriors of questions and problems to their respective priors
        :return: None, modify the database in place
        """
        db = self.db
        problem_cursor = db.problems.find()
        question_cursor = db.questions.find()
        for problem in problem_cursor:
            problem['posterior'] = problem['prior']
            db.problems.update({'_id': problem['_id']}, problem)
        # Reset the priors of the questions to take into account new priors on problems
        questions.reset_priors(self.db)
        # Set posterior = prior
        for question in question_cursor:
            question['posterior'] = question['prior']
            db.questions.update({'_id': question['_id']}, question)

    def adjust_question_posteriors(self, responses_known_so_far, most_likely_problems):
        """ Adjust the posteriors of the questions
        :param responses_known_so_far: Dictionary mapping hash values of questions and the response
        :param most_likely_problems: The most likely problems returned by Jenks/k-means
        :return: None, update db in place
        """
        # Update the posteriors of questions
        questions.adjust_posteriors(self.db, responses_known_so_far, most_likely_problems)

    def adjust_problem_posteriors(self, question, response, confidence):
        """ Adjust the posteriors of problems
        :param question: Dictionary of question whose posterior to adjust
        :param response: The response of user
        :param confidence: The confidence level of the user
        :return: None, update db in place
        """
        # Adjust the posteriors of the problems
        problems.adjust_posteriors(self.db, question, response, confidence)

    def ask_question(self, most_likely_questions):
        """ Ask a question and return the question, response and confidence level
        :param most_likely_questions: Most likely set of questions as obtained from Jenks
        :return: The dictionary of the question asked and the response given by the user
        """
        db = self.db
        count = db.questions.find().count()
        while count < 1:
            print('No questions in database!')
            training.train(db)
            count = db.questions.find().count()
        # Check to see if there is a question with non-zero posterior
        cursor = db.questions.find()
        m = 0
        for item in cursor:
            if item['posterior'] > 0:
                m += 1
                break
        if not m:
            return
        # Get list of questions with the highest posterior values by using k-means/Jenks
        most_likely_hash = set([item['hash'] for item in most_likely_questions])
        # Print the most helpful questions
        most_likely_names = set([item['name'] for item in most_likely_questions])
        print('Most helpful questions: ')
        questions.print_set(most_likely_names)
        question = questions.sample(db, 'posterior', most_likely_hash)
        while question is None:
            training.train(db)
            question = questions.sample(db, 'posterior')
        while True:
            try:
                response = eval(input(question['name']))
                try:
                    confidence = eval(input('Confidence in your answer (default 0.99) : '))
                except (NameError, ValueError, SyntaxError):
                    confidence = 0.95
                break
            except (SyntaxError, NameError, ValueError):
                helper.error_one_zero()
        return question, response, confidence

    def get_feedback(self, most_likely):
        """ Query the user if the correct problem was in this set
        :param most_likely: The most_likely set of problems
        :return:
        """
        while True:
            try:
                correct = eval(input('Are you happy with the result? (0/1) '))
                break
            except (NameError, SyntaxError, ValueError):
                helper.error_one_zero()
        most_likely_hash = [item['hash'] for item in most_likely]
        if correct:
            # Correct answer, increase count of each problem in this set
            list([problems.increment(self.db, x) for x in most_likely_hash])
        else:
            # Incorrect problem
            correct = sepquestions.get_correct_problem(self.db)
            for hash_value in most_likely_hash:
                if hash_value != correct['hash']:
                    wrong = self.db.problems.find_one({'hash': hash_value})
                    sepquestions.ask_separating_question(self.db, wrong, correct)
            # Increment the prior for the correct problem and set its posterior equal to prior
            problems.increment(self.db, correct['hash'])

    def query_backup(self):
        """ Query whether to backup the database
        :return:
        """
        db = self.db
        while True:
            try:
                response = eval(input('\nBackup database (0/1)? '))
                break
            except (NameError, SyntaxError, ValueError):
                helper.error_one_zero()
        if response:
            database.dump_db()

    def query_gvf_question(self):
        """ Query goodness of fit value for the questions from user
        :return: The dictionary items of the most likely questions
        """
        try:
            question_gvf = eval(input('Goodness of fit of questions (default = 0.6) '))
        except (NameError, SyntaxError):
            question_gvf = 0.6
        most_likely_questions = self.fit_posteriors('questions', question_gvf)
        return most_likely_questions

    def query_gvf_problems(self):
        """ Query goodness of fit value for the problems from user
        :return: The dictionary items of the most likely problems
        """
        try:
            problem_gvf = eval(input('Goodness of fit of problems (default = 0.8) '))
        except (NameError, SyntaxError):
            problem_gvf = 0.8
        most_likely_problems = self.fit_posteriors('problems', problem_gvf)
        return most_likely_problems

    def control_prediction(self):
        """ Control flow of questions
        :return: None, just control the flow of prediction
        """
        m = questions.max_posterior(self.db)
        continue_response = 1
        most_likely_problems = list()
        responses_known_so_far = dict()
        print('Current total entropy = %0.2f' % problems.get_entropy(self.db))
        problems.plot_posteriors(self.db)
        while m > 0 and continue_response:
            # Get the most_likely questions
            most_likely_questions = self.query_gvf_question()
            # Ask a question and get response from user
            question, response, confidence = self.ask_question(most_likely_questions)
            if question is None:
                break
            # Update the history_of_responses dictionary item
            responses_known_so_far[question['hash']] = (response, confidence)
            # Adjust the posteriors of the problems
            self.adjust_problem_posteriors(question, response, confidence)
            # Get the most likely set of problems
            most_likely_problems = self.query_gvf_problems()
            # Update the posteriors of the questions
            self.adjust_question_posteriors(responses_known_so_far, most_likely_problems)
            # Print the contents of the database
            self.print_table()
            # Print the most likely problems
            most_likely_problem_names = set([item['name'] for item in most_likely_problems])
            print('Popular problems that match your criteria:')
            problems.print_set(most_likely_problem_names)
            # Print the current entropy level of the distribution
            print(('Current total entropy = %0.2f' % problems.get_entropy(self.db)))
            # Plot problem posteriors
            problems.plot_posteriors(self.db)
            # Update the maximum posterior value
            m = questions.max_posterior(self.db)
            # Query whether to continue
            while True:
                try:
                    continue_response = eval(input('Ask more questions? (0/1) '))
                    break
                except (NameError, ValueError, SyntaxError):
                    helper.error_one_zero()
        if most_likely_problems:
            self.get_feedback(most_likely_problems)
        star = '*'*70
        print(star)

    def fit_posteriors(self, document, desired_gvf=0.8):
        """ Cluster the posteriors using Jenks Natural Breaks algorithm
        :param document: document in {problems, questions}
        :param desired_gvf: A number between [0, 1] showing goodness of fit
        :return: A list of the dictionary of the most likely problems/questions
        """
        # gvf denotes the goodness of fit, n denotes the number of classes in Jenks/k-means
        gvf = 0.0
        n = 0
        if document == 'problems':
            cursor = self.db.problems.find()
        elif document == 'questions':
            cursor = self.db.questions.find()
        else:
            return
        posteriors = list()
        i = 0
        idx_to_hash_name = dict()
        for item in cursor:
            posteriors.append(float(item['posterior']))
            idx_to_hash_name[i] = item
            i += 1
        array = np.array(posteriors)
        while gvf < desired_gvf:
            # Keep increasing n till gvf is at least the desired_gvf
            gvf = natural_break.gvf(array, n)
            n += 1
        centers = jenks(array, n)
        most_likely = list()
        for i in range(len(posteriors)):
            d = [(abs(posteriors[i] - centers[k]), k) for k in range(len(centers))]
            d.sort()
            if d[0][1] == len(centers) - 1:
                most_likely.append(idx_to_hash_name[i])
        return most_likely

    def add_problem(self):
        """ Add a problem to the database and query for YES questions and NO questions
        :return: None, update database in place
        """
        problem_name = helper.strip(input('Enter a problem: '))
        if not problem_name:
            return
        problem_hash = helper.get_hash(problem_name)
        problem = self.db.problems.find_one({'hash': problem_hash})
        # Print question list, ask what questions it has yes for an answer and no for an answer
        question_idx_to_id = questions.print_list(self.db)
        while True:
            try:
                yes_list = eval(input('Enter numbers of questions to which it answers YES: '))
                if type(yes_list) is int:
                    yes_qid_list = question_idx_to_id(yes_list)
                else:
                    yes_qid_list = [question_idx_to_id[x] for x in yes_list]
                break
            except (NameError, ValueError, SyntaxError):
                helper.error_comma()
            except KeyError:
                helper.error_key()
        pos_questions = list()
        neg_questions = list()
        for qid in yes_qid_list:
            question = self.db.questions.find_one({'_id': qid})
            pos_problems = question['posproblems']
            pos_problems.append(problem_hash)
            neg_problems = question['negproblems']
            neg_problems = [x for x in neg_problems if x != problem_hash]
            self.db.questions.update({'_id': question['_id']}, {
                '$set': {'posproblems': pos_problems, 'negproblems': neg_problems}
            })
            pos_questions.append(question['hash'])
        while True:
            try:
                no_list = eval(input('Enter numbers of questions to which it answers NO: '))
                if type(no_list) is int:
                    no_qid_list = question_idx_to_id[no_list]
                else:
                    no_qid_list = [question_idx_to_id[x] for x in no_list]
                break
            except ValueError:
                helper.error_comma()
            except KeyError:
                helper.error_key()
        for qid in no_qid_list:
            question = self.db.questions.find_one({'_id': qid})
            neg_problems = question['negproblems']
            neg_problems.append(problem_hash)
            pos_problems = question['posproblems']
            pos_problems = [x for x in pos_problems if x != problem_hash]
            self.db.questions.update({'_id': question['_id']}, {
                '$set': {'posproblems': pos_problems, 'negproblems': neg_problems}
            })
            neg_questions.append(question['hash'])
        # Finally update the problem or insert it
        if problem is None:
            # New problem
            prior = 1
            posterior = 1
            d = {'name': problem_name, 'hash': problem_hash, 'prior': prior,
                 'posterior': posterior, 'posquestions': pos_questions,
                 'negquestions': neg_questions}
            self.db.problems.insert_one(d)
        else:
            # Problem already existed, append pos_questions and neg_questions to it
            positive_list = problem['posquestions']
            negative_list = problem['negquestions']
            positive_set = set(positive_list)
            negative_set = set(negative_list)
            for i in range(len(pos_questions)):
                if pos_questions[i] not in positive_set:
                    positive_list.append(pos_questions[i])
            for i in range(len(neg_questions)):
                if neg_questions[i] not in negative_set:
                    negative_list.append(neg_questions[i])
            self.db.problems.update({'_id': problem['_id']}, {
                '$set': {'posquestions': positive_list, 'negquestions': negative_list}
            })

    def add_question(self):
        """ Add a question to the database and query for problems with YES answers and NO answers
        :return: None, update database in place
        """
        question_name = helper.strip(input('Enter a question: '))
        if not question_name:
            return
        question_hash = helper.get_hash(question_name)
        question = self.db.questions.find_one({'hash': question_hash})
        # Print problem list, ask what problems it has YES and NO answers for
        problem_idx_to_id = problems.print_list(self.db)
        while True:
            try:
                yes_list = eval(input('Enter numbers of problems that have a YES answer to this question: '))
                if type(yes_list) is int:
                    yes_pid_list = problem_idx_to_id[yes_list]
                else:
                    yes_pid_list = [problem_idx_to_id[x] for x in yes_list]
                break
            except ValueError:
                helper.error_comma()
            except KeyError:
                helper.error_key()
        pos_problems = list()
        neg_problems = list()
        for pid in yes_pid_list:
            problem = self.db.problems.find_one({'_id': pid})
            pos_questions = problem['posquestions']
            pos_questions.append(question_hash)
            neg_questions = problem['negquestions']
            neg_questions = [x for x in neg_questions if x != question_hash]
            self.db.problems.update({'_id': problem['_id']}, {
                '$set': {'posquestions': pos_questions, 'negquestions': neg_questions}
            })
            pos_problems.append(problem['hash'])
        while True:
            try:
                no_list = eval(input('Enter numbers of problems that have a NO answer to this question: '))
                if type(no_list) is int:
                    no_pid_list = problem_idx_to_id[no_list]
                else:
                    no_pid_list = [problem_idx_to_id[x] for x in no_list]
                break
            except ValueError:
                helper.error_comma()
            except KeyError:
                helper.error_key()
        for pid in no_pid_list:
            problem = self.db.problems.find_one({'_id': pid})
            neg_questions = problem['negquestions']
            neg_questions.append(question_hash)
            pos_questions = problem['posquestions']
            pos_questions = [x for x in pos_questions if x != question_hash]
            self.db.problems.update({'_id': problem['_id']}, {
                '$set': {'posquestions': pos_questions, 'negquestions': neg_questions}
            })
            neg_problems.append(problem['hash'])
        # Finally update the question or insert it
        if question is None:
            # New question
            table = 'problems'
            item_property = 'prior'
            # Get mass of YES problems
            pos_mass, neg_mass = 0, 0
            if pos_problems:
                q_positive_problem_mass = [helper.mass(self.db, table, x, item_property) for x in pos_problems]
                pos_mass = reduce(lambda x, y: x + y, q_positive_problem_mass)
            # Get mass of NO problems
            if neg_problems:
                q_negative_problem_mass = [helper.mass(self.db, table, x, item_property) for x in neg_problems]
                neg_mass = reduce(lambda x, y: x + y, q_negative_problem_mass)
            # Define prior as total mass of pairs separated
            prior = pos_mass * neg_mass
            posterior = prior
            d = {'name': question_name, 'hash': question_hash, 'prior': prior,
             'posterior': posterior, 'posproblems': pos_problems,
             'negproblems': neg_problems}
            self.db.questions.insert_one(d)
        else:
            # Question already existed, append pos_problems and neg_problems to it
            positive_list = question['posproblems']
            negative_list = question['negproblems']
            positive_set = set(positive_list)
            negative_set = set(negative_list)
            for i in range(len(pos_problems)):
                if pos_problems[i] not in positive_set:
                    positive_list.append(pos_problems[i])
            for i in range(len(neg_problems)):
                if neg_problems[i] not in negative_set:
                    negative_list.append(neg_problems[i])
            table = 'problems'
            item_property = 'prior'
            # Get mass of YES problems
            pos_mass, neg_mass = 0, 0
            if positive_list:
                q_positive_problem_mass = [helper.mass(self.db, table, x, item_property) for x in positive_list]
                pos_mass = reduce(lambda x, y: x + y, q_positive_problem_mass)
            # Get mass of NO problems
            if negative_list:
                q_negative_problem_mass = [helper.mass(self.db, table, x, item_property) for x in negative_list]
                neg_mass = reduce(lambda x, y: x + y, q_negative_problem_mass)
            # Define prior as total mass of pairs separated
            prior = pos_mass * neg_mass
            posterior = prior
            self.db.questions.update({'_id': question['_id']}, {
                '$set': {'prior': prior, 'posterior': posterior,
                         'posproblems': positive_list, 'negproblems': negative_list}
            })

    def download_arxiv(self, flag, max_results, keywords):
        """ Download papers from arxiv, integration with word2vec
        :param flag: flag in {Yes, No}, if Yes downloads pdfs too
        :return: None, update db in place
        """
        try:
            keywords = keywords.strip().split(",")
            arxiv.download(self.db, flag, max_results, keywords)
        except KeyboardInterrupt:
            self.query_backup()

    def count_papers(self):
        """ Count the number of papers in self.db.papers
        :return: Number of papers
        """
        cursor = self.db.papers.find()
        return cursor.count()

    def make_uniform(self, n):
        """ Make the priors of all the problems in the database uniform
        :param n: Set the priors to n
        :return: None, update DB in place
        """
        cursor = self.db.problems.find()
        for item in cursor:
            item['prior'] = n
            self.db.problems.update({'hash': item['hash']}, item)

    def view_problem_structure(self):
        """ View the YES questions and NO questions of a problem
        :return: None
        """
        problems.view_questions(self.db)

    def view_question_structure(self):
        """ View the YES problems and NO problems of a question
        :return:None
        """
        questions.view_problems(self.db)

    def get_downloaded_keywords(self):
        """ Get the list of keywords downloaded so far
        :return: Set of downloaded keywords
        """
        keywords = dict()
        cursor = self.db.papers.find()
        for item in cursor:
            if item['keyword'] in keywords:
                keywords[item['keyword']] += 1
            else:
                keywords[item['keyword']] = 1
        return keywords

    def adjust_problem_priors_from_summary(self, data_set, doc2vec_model_path, classifier_path, summary):
        """ Adjust the priors of problems from the summary
        :param summary: Summary of text
        :param model_name: Name of the model
        :return: None
        """
        distribution = model.classify(data_set, doc2vec_model_path, classifier_path, summary)
        for key in distribution:
            item = self.db.problems.find_one({'name': key})
            item['prior'] *= distribution[key]
            self.db.problems.update({'hash': item['hash']}, item)
        p = np.array([distribution[key] for key in distribution])
        e = entropy(p)
        print('Entropy = ', e)


    def adjust_question_priors_from_summary(self, summary, model_name):
        """ Adjust the priors of problems from the summary
        :param summary: Word tokens of summary
        :param model_name: Name of the model
        :return: None
        """
        model = gensim.models.Doc2Vec.load(model_name)
        cursor = self.db.questions.find()
        for item in cursor:
            # Tokenize the question name
            question = []
            question_tokens = word_tokenize(item['name'].strip().encode("utf-8"))
            # Remove non-alpha characters from the words
            for w in question_tokens:
                scrunched = model.scrunch(w)
                if scrunched:
                    question.append(scrunched)
            # Remove short words, convert to lower case
            question = model.small_words(question)
            # Remove stop words
            question = model.remove_stop(question)
            similarity = model.n_similarity(summary, question)
            print((item['name'] + ': ' + str(similarity)))
            item['prior'] *= (1 + similarity)/2
            self.db.questions.update({'hash': item['hash']}, item)

    def clean_database(self):
        """ Clean database of abstracts that contain physics words, short abstracts and remove crap
        :return: None, clean db
        """
        cursor = self.db.papers.find(no_cursor_timeout=True)
        stop_words = {'physics', 'quantum', 'hep', 'topology', 'optics', 'electrodynamics', 'Doppler', 'Cookies'}
        counter = 0
        print('Cleaning up physics stuff...')
        for item in cursor:
            words = set(item['abstract'].strip().split())
            if words & stop_words:
                self.db.papers.delete_one({'_id': item['_id']})
                counter += 1
            elif len(sent_tokenize(item['abstract'])) < 3:
                # Abstract should have at least 2 sentences
                print('----------------Abstract-----------')
                print((item['abstract']))
                self.db.papers.delete_one({'hash': item['hash']})
                counter += 1
        # Close cursor
        cursor.close()
        print(('Deleted %d entries' % counter))
        # Remove non-alpha characters from the words
        cursor = self.db.papers.find(no_cursor_timeout=True)
        # Clean abstracts for now
        print('Cleaning up abstracts...')
        for item in cursor:
            helper.clean_text(self.db, 0, item['hash'])
        cursor.close()

    def dump_data_to_txt(self, data_path):
        """ Dump the papers to a folder with folders inside keywords
        :param data_path: path to dump to
        :return:
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        keywords = self.get_downloaded_keywords()
        for keyword in keywords:
            print(('Writing abstracts for ', keyword))
            keyword_path = data_path + '/' + keyword
            if not os.path.exists(keyword_path):
                os.makedirs(keyword_path)
            cursor = self.db.papers.find({'keyword': keyword}, no_cursor_timeout=True)
            count = 0
            for item in cursor:
                file_name = keyword_path + '/' + 'abstract_' + str(count)
                f = open(file_name, 'wb')
                f.write(item['abstract'].encode('utf-8'))
                f.close()
                count += 1
            cursor.close()

    def import_problems_from_txt(self, data_path):
        """ Given a data path import problems from the sub folders
        :return: None, update data base
        """
        problem_list = os.listdir(data_path)
        for problem in problem_list:
            hash_value = helper.get_hash(problem)
            item = self.db.problems.find_one({'hash': hash_value})
            if item is None:
                prior = 1
                posterior = 1
                posquestions = list()
                negquestions = list()
                d = {'name': problem, 'hash': hash_value, 'prior': prior,
                     'posterior': posterior, 'posquestions': posquestions,
                     'negquestions': negquestions}
                self.db.problems.insert(d)
