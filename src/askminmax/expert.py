import helper
import database
import problems
import questions
import sepquestions
import training
import arxiv
import cluster


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


    def adjustposteriors(self, question, response, confidence):
        ''' Adjust posteriors of problems and questions
        :param question: Dictionary of question whose posterior to adjust
        :param response: 0 or 1, asnwer to this question
        :param confidence: A number between 0 and 1 showing confidence in the user's answer
        :return: None, update db in place
        '''
        db = self.db
        # Adjust the posteriors of the problems based on response and confidence level
        problems.adjustposteriors(db, question, response, confidence)
        # Since the mass on problems has changed, posteriors of questions need to be updated
        questions.adjustposteriors(db)
        # Set posterior of this question to 0, essentially it should not be asked again
        question['posterior'] = 0
        db.questions.update({'_id': question['_id']}, question)


    def askquestions(self, n):
        ''' Predict by asking n questions
        :param n: The number of questions to ask
        :return: None, update db in place
        '''
        db = self.db
        count = db.questions.find().count()
        while count < 1:
            print 'No questions in database!'
            training.train(db)
            count = db.questions.find().count()
        for i in xrange(n):
            cursor = db.questions.find()
            m = 0
            for item in cursor:
                if item['posterior'] > 0:
                    m += 1
            if not m:
                return
            question = questions.sample(db, 'posterior')
            while question is None:
                training.train(db)
                question = questions.sample(db, 'posterior')
            while True:
                confidence = 1.0
                try:
                    response = int(raw_input(question['name']))
                    confidence = float(raw_input('Confidence in your answer [0, 1]: '))
                    break
                except ValueError:
                    helper.erroronezero()
            self.adjustposteriors(question, response, confidence)


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
            self.askquestions(1)
            m = questions.maxposterior(db)
            while True:
                try:
                    response = int(raw_input('Ask more questions? (0/1) '))
                    break
                except ValueError:
                    helper.erroronezero()
        while True:
            try:
                flag = int(raw_input('Predict single problem? (0/1) '))
                break
            except ValueError:
                helper.erroronezero()
        if flag:
            self.predictsingle()
        else:
            while True:
                try:
                    n = int(raw_input('Maximum size of set? '))
                    break
                except ValueError:
                    helper.errornumber()
            self.predictset(n)


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