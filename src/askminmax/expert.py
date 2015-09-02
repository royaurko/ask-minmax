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
        # Reset the priors of the questions to take into account new priors on problems
        questions.resetpriors(self.db)
        # Set posterior = prior
        for question in question_cursor:
            question['posterior'] = question['prior']
            db.questions.update({'_id': question['_id']}, question)


    def adjustposteriors(self, question, response, confidence):
        ''' Adjust posteriors of problems and questions
        :param question: Dictionary of question whose posterior to adjust
        :param response: The response of user
        :param confidence: The confidence level of the user
        :param most_likely_problems_hash: Optional, if given question posteriors are updated depending on it
        :return: None, update db in place
        '''
        db = self.db
        # Adjust the posteriors of the problems
        problems.adjustposteriors(db, question, response, confidence)
        # Update the posteriors of questions
        questions.adjustposteriors(db)
        # Set posterior of this question to 0, essentially it should not be asked again
        # question['posterior'] = 0
        # db.questions.update({'_id': question['_id']}, question)


    def askquestion(self, most_likely_questions):
        ''' Ask a question and update posteriors by calling adjust posteriors
        :param most_likely_questions: Most likely set of questions as obtained from Jenks
        :return: The dictionary of the question asked
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
        # Get list of questions with the highest posterior values by using k-means/Jenks
        most_likely_hash = set([item['hash'] for item in most_likely_questions])
        # Print the most helpful questions
        most_likely_names = set([item['name'] for item in most_likely_questions])
        print 'Most helpful questions: '
        questions.printset(most_likely_names)
        question = questions.sample(db, 'posterior', most_likely_hash)
        while question is None:
            training.train(db)
            question = questions.sample(db, 'posterior')
        while True:
            try:
                response = int(raw_input(question['name']))
                try:
                    confidence = float(raw_input('Confidence in your answer (default 0.99) : '))
                except ValueError:
                    confidence = 0.95
                break
            except ValueError:
                helper.erroronezero()
        # Adjust the posteriors of the problems and questions
        self.adjustposteriors(question, response, confidence)
        self.printtable()
        return question


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
        :param most_likely: The most_likely set of problems
        :return:
        '''
        while True:
            try:
                correct = int(raw_input('Are you happy with the result? (0/1) '))
                break
            except ValueError:
                helper.erroronezero()
        most_likely_hash = [item['hash'] for item in most_likely]
        if correct:
            # Correct answer, increase count of each problem in this set
            map(lambda x: problems.increment(self.db, x), most_likely_hash)
        else:
            # Incorrect problem
            correct = sepquestions.getcorrectproblem(self.db)
            for hashval in most_likely_hash:
                if hashval != correct['hash']:
                    wrong = self.db.problems.find_one({'hash': hashval})
                    sepquestions.separatingquestion(self.db, wrong, correct)
            # Increment the prior for the correct problem and set its posterior equal to prior
            problems.increment(self.db, correct['hash'])


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
        most_likely_problems, asked_questions = set(), set()
        print 'Current total entropy = %0.2f' % problems.getentropy(self.db)
        while m > 0 and response:
            most_likely_questions = self.fitposteriors('questions', 0.6)
            most_likely_questions = [q for q in most_likely_questions if q['hash'] not in asked_questions]
            question = self.askquestion(most_likely_questions)
            if question is None:
                break
            asked_questions.add(question['hash'])
            try:
                gvf = float(raw_input('Goodness of fit (default = 0.8) '))
            except ValueError:
                gvf = 0.8
            most_likely_problems = self.fitposteriors('problems', gvf)
            most_likely_problem_names = [item['name'] for item in most_likely_problems]
            print 'Popular problems that match your criteria:'
            problems.printset(most_likely_problem_names)
            print 'Current total entropy = %0.2f' % problems.getentropy(self.db)
            m = questions.maxposterior(db)
            while True:
                try:
                    response = int(raw_input('Ask more questions? (0/1) '))
                    break
                except ValueError:
                    helper.erroronezero()
        if most_likely_problems:
            self.getfeedback(most_likely_problems)
        star = '*'*70
        print star


    def fitposteriors(self, document, desired_gvf=0.8):
        ''' Try to cluster the posteriors using Jenks Natural Breaks algorithm
        :param document: document in {problems, questions}
        :param desired_gvf: A number between [0, 1] showing goodness of fit
        :return: A list of the dictionary of the most likely problems/questions
        '''
        gvf = 0.0
        nclasses = 0
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
            # Keep increasing nclasses till Goodness of fit is atleast 0.8 say
            gvf = natural_break.gvf(array, nclasses)
            nclasses += 1
        centers = jenks(array, nclasses)
        most_likely = list()
        for i in xrange(len(posteriors)):
            d = [(abs(posteriors[i] - centers[k]), k) for k in xrange(len(centers))]
            d.sort()
            if d[0][1] == len(centers) - 1:
                most_likely.append(idx_to_hash_name[i])
        return most_likely


    def addproblem(self):
        ''' Add a problem to the database and query for YES questions and NO questions
        :return: None, update database in place
        '''
        pname = helper.strip(raw_input('Enter a problem: '))
        if not pname:
            return
        phash_val = helper.gethashval(pname)
        problem = self.db.problems.find_one({'hash': phash_val})
        # Print question list, ask what questions it has yes for an answer and no for an answer
        question_idx_to_id = questions.printlist(self.db)
        while True:
            try:
                yes_list = raw_input('Enter numbers of questions to which it answers YES: ')
                yes_list = map(int, yes_list.strip().split())
                yes_qid_list = [question_idx_to_id[x] for x in yes_list]
                break
            except ValueError:
                helper.errorspaces()
            except KeyError:
                helper.errorkey()
        pos_questions = list()
        neg_questions = list()
        for qid in yes_qid_list:
            question = self.db.questions.find_one({'_id': qid})
            pos_problems = question['posproblems']
            pos_problems.append(phash_val)
            neg_problems = question['negproblems']
            neg_problems = [x for x in neg_problems if x != phash_val]
            self.db.questions.update({'_id': question['_id']}, {
                '$set': {'posproblems': pos_problems, 'negproblems': neg_problems}
            })
            pos_questions.append(question['hash'])
        while True:
            try:
                no_list = raw_input('Enter numbers of questions to which it answers NO: ')
                no_list = map(int, no_list.strip().split())
                no_qid_list = [question_idx_to_id[x] for x in no_list]
                break
            except ValueError:
                helper.errorspaces()
            except KeyError:
                helper.errorkey()
        for qid in no_qid_list:
            question = self.db.questions.find_one({'_id': qid})
            neg_problems = question['negproblems']
            neg_problems.append(phash_val)
            pos_problems = question['posproblems']
            pos_problems = [x for x in pos_problems if x != phash_val]
            self.db.questions.update({'_id': question['_id']}, {
                '$set': {'posproblems': pos_problems, 'negproblems': neg_problems}
            })
            neg_questions.append(question['hash'])
        # Finally update the problem or insert it
        if problem is None:
            # New problem
            prior = 1
            posterior = 1
            d = {'name': pname, 'hash': phash_val, 'prior': prior,
             'posterior': posterior, 'posquestions': pos_questions,
             'negquestions': neg_questions}
            self.db.problems.insert_one(d)
        else:
            # Problem already existed, append pos_questions and neg_questions to it
            plist = problem['posquestions']
            nlist = problem['negquestions']
            plist_set = set(plist)
            nlist_set = set(nlist)
            for i in xrange(len(pos_questions)):
                if pos_questions[i] not in plist_set:
                    plist.append(pos_questions[i])
            for i in xrange(len(neg_questions)):
                if neg_questions[i] not in nlist_set:
                    nlist.append(neg_questions[i])
            self.db.problems.update({'_id': problem['_id']}, {
                '$set':{'posquestions': plist, 'negquestions': nlist}
            })


    def addquestion(self):
        ''' Add a question to the database and query for problems with YES answers and NO answers
        :return: None, update database in place
        '''
        qname = helper.strip(raw_input('Enter a question: '))
        if not qname:
            return
        qhash_val = helper.gethashval(qname)
        question = self.db.questions.find_one({'hash': qhash_val})
        # Print problem list, ask what problems it has yes answers and no answers for
        problem_idx_to_id = problems.printlist(self.db)
        while True:
            try:
                yes_list = raw_input('Enter numbers of problems that have a YES answer to this question: ')
                yes_list = map(int, yes_list.strip().split())
                yes_pid_list = [problem_idx_to_id[x] for x in yes_list]
                break
            except ValueError:
                helper.errorspaces()
            except KeyError:
                helper.errorkey()
        pos_problems = list()
        neg_problems = list()
        for pid in yes_pid_list:
            problem = self.db.problems.find_one({'_id': pid})
            pos_questions = problem['posquestions']
            pos_questions.append(qhash_val)
            neg_questions = problem['negquestions']
            neg_questions = [x for x in neg_questions if x != qhash_val]
            self.db.problems.update({'_id': problem['_id']}, {
                '$set': {'posquestions': pos_questions, 'negquestions': neg_questions}
            })
            pos_problems.append(problem['hash'])
        while True:
            try:
                no_list = raw_input('Enter numbers of problems that have a NO answer to this question: ')
                no_list = map(int, no_list.strip().split())
                no_pid_list = [problem_idx_to_id[x] for x in no_list]
                break
            except ValueError:
                helper.errorspaces()
            except KeyError:
                helper.errorkey()
        for pid in no_pid_list:
            problem = self.db.problems.find_one({'_id': pid})
            neg_questions = problem['negquestions']
            neg_questions.append(qhash_val)
            pos_questions = problem['posquestions']
            pos_questions = [x for x in pos_questions if x != qhash_val]
            self.db.problems.update({'_id': problem['_id']}, {
                '$set': {'posquestions': pos_questions, 'negquestions': neg_questions}
            })
            neg_problems.append(problem['hash'])
        # Finally update the question or insert it
        if question is None:
            # New question
            table = 'problems'
            property = 'prior'
            # Get mass of YES problems
            pos_mass, neg_mass = 0, 0
            if pos_problems:
                q_posproblem_mass = map(lambda x: helper.mass(self.db, table, x, property), pos_problems)
                pos_mass = reduce(lambda x, y: x + y, q_posproblem_mass)
            # Get mass of NO problems
            if neg_problems:
                q_negproblem_mass = map(lambda x: helper.mass(self.db, table, x, property), neg_problems)
                neg_mass = reduce(lambda x, y: x + y, q_negproblem_mass)
            # Define prior as total mass of pairs separated
            prior = pos_mass * neg_mass
            posterior = prior
            loglikelihood = 0.0
            d = {'name': qname, 'hash': qhash_val, 'prior': prior,
             'posterior': posterior, 'posproblems': pos_problems,
             'negproblems': neg_problems, 'loglikelihood': loglikelihood}
            self.db.questions.insert_one(d)
        else:
            # Question already existed, append pos_problems and neg_problems to it
            plist = question['posproblems']
            nlist = question['negproblems']
            plist_set = set(plist)
            nlist_set = set(nlist)
            for i in xrange(len(pos_problems)):
                if pos_problems[i] not in plist_set:
                    plist.append(pos_problems[i])
            for i in xrange(len(neg_problems)):
                if neg_problems[i] not in nlist_set:
                    nlist.append(neg_problems[i])
            table = 'problems'
            property = 'prior'
            # Get mass of YES problems
            pos_mass, neg_mass = 0, 0
            if plist:
                q_posproblem_mass = map(lambda x: helper.mass(self.db, table, x, property), plist)
                pos_mass = reduce(lambda x, y: x + y, q_posproblem_mass)
            # Get mass of NO problems
            if nlist:
                q_negproblem_mass = map(lambda x: helper.mass(self.db, table, x, property), nlist)
                neg_mass = reduce(lambda x, y: x + y, q_negproblem_mass)
            # Define prior as total mass of pairs separated
            prior = pos_mass * neg_mass
            posterior = prior
            self.db.questions.update({'_id': question['_id']}, {
                '$set': {'prior': prior, 'posterior': posterior, 'posproblems': plist, 'negproblems': nlist}
            })


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


    def makeuniform(self):
        ''' Make the priors of all the problems in the database uniform
        :return: None, update DB in place
        '''
        while True:
            try:
                n = int(raw_input('Make prior for every problem = '))
                break
            except ValueError:
                helper.errornumber()
        cursor = self.db.problems.find()
        for item in cursor:
            item['prior'] = n
            self.db.problems.update({'_id': item['_id']}, item)



    def cluster(self):
        ''' Run the Word2Vec model on the papers and k-means
        :return: None for now
        '''
        cluster.clusterTests(self.db)