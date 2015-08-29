import helper
import database
import problems
import questions
import sepquestions
import training
import arxiv
import cluster
import natural_break
import numpy as np
from jenks import jenks


class Expert(object):
    def __init__(self):
        ''' Constructor for Expert class
        :return: None
        '''
        while True:
            try:
                build_question = 'Build new database (0/1)? '
                response = int(raw_input(build_question))
                break
            except ValueError:
                helper.erroronezero()
        client, db = database.initializedb()
        if response:
            problems.query(db)
            training.train(db)
        else:
            while True:
                try:
                    recover_question = 'Recover a database from bson file? '
                    flag = int(raw_input(recover_question))
                    break
                except ValueError:
                    helper.erroronezero()
            if flag:
                db = database.recoverdb(client)
        # Set the expert instance database to db
        self.db = db


    def train(self):
        ''' Just call the train subroutine from training to learn separating questions
        :return: None, modify db in place
        '''
        training.train(self.db)


    def delete(self):
        ''' Allows you to delete a problem or a question from the database
        :return: None, modify database in place
        '''
        db = self.db
        problem_idx_to_id = problems.printlist(db)
        problems_list = raw_input('Enter indices of problems to delete separated by spaces: ')
        problems_list = map(int, problems_list.strip().split())
        for problem in problems_list:
            problems.delete(db, problem_idx_to_id[problem])
        question_idx_to_id = questions.printlist(db)
        questions_list = raw_input('Enter indices of questions to delete separated by spaces: ')
        questions_list = map(int, questions_list.strip().split())
        for question in questions_list:
            questions.delete(db, question_idx_to_id[question])
        print 'Modified database:'
        self.printtable()


    def printtable(self):
        ''' Print the current list of problems and questions with their priors and posteriors
        :return: None
        '''
        db = self.db
        problems.printlist(db)
        questions.printlist(db)


    def run(self):
        ''' Control the main program flow
        :return: None, modify db in place
        '''
        try:
            while True:
                # Reset the posteriors equal to prior before starting a prediction loop
                self.resetposteriors()
                # Print the table
                self.printtable()
                # Call controlprediction
                self.controlprediction()
                print 'Press [Ctrl] + c to exit'
        except KeyboardInterrupt:
            self.querybackup()


    def resetposteriors(self):
        ''' Reset posteriors of questions and problems to their respective priors
        :return: None, modify the database in place
        '''
        db = self.db
        problem_cursor = db.problems.find()
        question_cursor = db.questions.find()
        for problem in problem_cursor:
            problem['posterior'] = problem['prior']
            db.problems.update({'_id': problem['_id']}, problem)
        for question in question_cursor:
            question['posterior'] = question['prior']
            db.questions.update({'_id': question['_id']}, question)


    def adjustquestionposteriors(self, question):
        ''' Adjust posteriors of questions based on most_likely set
        :param question: Dictionary of question whose posterior to adjust
        :return: Name of most_likely set of problems
        '''
        db = self.db
        # Get the most_likely set of problems by calling on fit posteriors
        try:
            gvf = float(raw_input('Goodness of fit (default = 0.8) '))
        except ValueError:
            gvf = 0.8
        most_likely = self.fitposteriors(gvf)
        # Since the mass on problems has changed, posteriors of questions need to be updated
        questions.adjustposteriors(db, most_likely)
        # Set posterior of this question to 0, essentially it should not be asked again
        question['posterior'] = 0
        db.questions.update({'_id': question['_id']}, question)
        # Return the most likely set of problems (hash, name)
        return most_likely


    def askquestion(self):
        ''' Ask a question and update posteriors by calling adjust posteriors
        :return: Most_likely set of problem names
        '''
        db = self.db
        count = db.questions.find().count()
        while count < 1:
            print 'No questions in database!'
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
        question = questions.sample(db, 'posterior')
        while question is None:
            training.train(db)
            question = questions.sample(db, 'posterior')
        while True:
            try:
                response = int(raw_input(question['name']))
                try:
                    confidence = float(raw_input('Confidence in your answer (default 0.8) : '))
                except ValueError:
                    confidence = 0.8
                break
            except ValueError:
                helper.erroronezero()
        # Adjust the posteriors of the problems
        problems.adjustposteriors(db, question, response, confidence)
        # Adjust the posteriors of the questions
        most_likely = self.adjustquestionposteriors(question)
        self.printtable()
        return most_likely


    def predictsingle(self):
        ''' Predict a single problem by sampling once from problem posterior
        :return: None
        '''
        db = self.db
        problem = problems.sample(db, 'posterior')
        print problem['name']
        while True:
            try:
                response = int(raw_input('Is this the correct problem? (0/1)? '))
                break
            except ValueError:
                helper.erroronezero()
        if response:
            # Correct answer, increase prior of the correct problem and set posterior = prior
            problems.increment(db, problem['hash'])
        else:
            # Wrong answer, call subroutine for separating question
            sepquestions.separatingquestion(db, problem)


    def predictset(self, n):
        ''' Predict a set of problems by sampling n times from posterior
        :param n: Size of the set to predict
        :return: None
        '''
        db = self.db
        problem_hash = set()
        problem_name = set()
        for i in xrange(n):
            problem = problems.sample(db, 'posterior')
            problem_hash.add(problem['hash'])
            problem_name.add(problem['name'])
        problems.printset(problem_name)
        while True:
            try:
                set_question = 'Is the correct problem in this set? (0/1)'
                flag = int(raw_input(set_question))
                break
            except ValueError:
                helper.erroronezero()
        if flag:
            # Correct answer, increase count of each problem in the set
            map(lambda x: problems.increment(db, x), problem_hash)
        else:
            # Wrong answer, ask for a separating question for each problem in set
            for hashval in problem_hash:
                    problem = db.problems.find_one({'hash': hashval})
                    sepquestions.separatingquestion(db, problem)


    def getfeedback(self, most_likely):
        ''' Query the user if the correct problem was in this set
        :param most_likely:
        :return:
        '''
        while True:
            try:
                correct = int(raw_input('Are you happy with the result? (0/1) '))
                break
            except ValueError:
                helper.erroronezero()
        most_likely_hash = [item[0] for item in most_likely]
        if correct:
            # Correct answer, increase count of each problem in this set
            map(lambda x: problems.increment(self.db, x), most_likely_hash)
        else:
            # Incorrect problem
            correct, correct_hash = sepquestions.getcorrectproblem(self.db)
            for hashval in most_likely_hash:
                if hashval != correct_hash:
                    problem = self.db.problems.find_one({'hash': hashval})
                    sepquestions.separatingquestion(self.db, problem, correct, correct_hash)
            # Increment the prior for the correct problem and set its posterior equal to prior
            problems.increment(self.db, correct_hash)


    def querybackup(self):
        ''' Query whether to backup the database
        :return:
        '''
        db = self.db
        while True:
            try:
                response = int(raw_input('\nBackup database (0/1)? '))
                break
            except ValueError:
                helper.erroronezero()
        if response:
            database.dumpdb(db)


    def controlprediction(self):
        ''' Control flow of questions
        :return: None, just control the flow of prediction
        '''
        db = self.db
        m = questions.maxposterior(db)
        response = 1
        while m > 0 and response:
            most_likely = self.askquestion()
            most_likely_names = [item[1] for item in most_likely]
            m = questions.maxposterior(db)
            print 'Most likely set of problems:'
            problems.printset(most_likely_names)
            if len(most_likely_names) > 1:
                while True:
                    try:
                        response = int(raw_input('Refine set further? (0/1) '))
                        break
                    except ValueError:
                        helper.erroronezero()
        self.getfeedback(most_likely)
        star = '*'*70
        print star


    def fitposteriors(self, desired_gvf=0.8):
        ''' Try to cluster the posteriors using Jenks Natural Breaks algorithm
        :return: The hash value of the problems and their names as a tuple
        '''
        gvf = 0.0
        nclasses = 0
        cursor = self.db.problems.find()
        posteriors = list()
        i = 0
        idx_to_hash_name = dict()
        for item in cursor:
            posteriors.append(float(item['posterior']))
            idx_to_hash_name[i] = (item['hash'], item['name'])
            i += 1
        array = np.array(posteriors)
        while gvf < desired_gvf:
            # Keep increasing nclasses till Goodness of fit is atleast 0.8 say
            gvf = natural_break.gvf(array, nclasses)
            nclasses += 1
        centers = jenks(array, nclasses)
        most_likely = set()
        for i in xrange(len(posteriors)):
            d = [(abs(posteriors[i] - centers[k]), k) for k in xrange(len(centers))]
            d.sort()
            if d[0][1] == len(centers) - 1:
                most_likely.add(idx_to_hash_name[i])
        return most_likely


    def download(self, keywords):
        ''' Download papers from arxiv, integration with word2vec
        :return: None, update db in place
        '''
        try:
            keywords = keywords.strip().split(",")
            arxiv.download(self.db, keywords)
        except KeyboardInterrupt:
            self.querybackup()


    def countpapers(self):
        ''' Count the number of papers in self.db.papers
        :return: Number of papers
        '''
        cursor = self.db.papers.find()
        return cursor.count()


    def cluster(self):
        ''' Run the Word2Vec model on the papers and k-means
        :return: None for now
        '''
        cluster.clusterTests(self.db)