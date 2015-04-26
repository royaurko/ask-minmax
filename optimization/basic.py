import json
import random


def setprior(f):
    '''Set priors for problems, assumes the depth of the problem tree is at most 3'''
    domains = json.load(f)
    n = float(len(domains))
    p = 1/n
    priors = dict()
    for item in domains:
        n1 = len(domains[item])
        if n1 > 0:
            p1 = p/n1
            for it1 in domains[item]:
                n2 = len(domains[item][it1])
                if n2 > 0:
                    p2 = p1/n2
                    for it2 in domains[item][it1]:
                        key = item + ' : ' + it1 + ' : ' + it2
                        priors[key] = p2
                else:
                    key = item + ' : ' + it1
                    priors[key] = p1
        else:
            priors[item] = p
    return priors


def sample(d):
    '''Sample keys from the dictionary proportional to the value'''
    weight = sum(d.itervalues())
    r = random.uniform(0, weight)
    s = 0.0
    for k, w in prior.iteritems():
        s += w
        if r < s:
            return k
    return k


def trials(n, q):
    '''n is the number of trials, q is the set of questions known till now'''
    for i in range(n):
        print sample(prior)
        flag = int(raw_input('Is this the correct answer (0/1)? '))
        if flag:
            break
        q.add(raw_input('What is a separating question?\n'))
    return q


if __name__ == '__main__':
    fname = raw_input('Domain file: ')
    f = open(fname, 'r')
    prior = setprior(f)
    q = set()
    n = 10
    q = trials(n, q)
