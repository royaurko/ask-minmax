# Contains all the functions responsible for maintaining the database

import subprocess
import os
from pymongo import MongoClient


def initializedb():
    '''Initialize the database with problems and questions'''
    client = MongoClient()
    db = client.db
    db.problems.drop()
    db.questions.drop()
    return db


def dumpdb(db):
    '''Dump database to bson file for use later'''
    try:
        path = raw_input('Name of folder to dump BSON files to? ')
        dirpath = os.path.dirname(os.path.abspath(__file__))
        dirpath = dirpath[:-3]
        path = dirpath + path
        cmd = 'mongodump'
        output = subprocess.call([cmd, '-o', path])
        print output
    except:
        print 'Error backing up database!'


def recoverdb():
    '''Recover a db from a bson dump'''
    client = MongoClient()
    client.drop_database('db')
    cmd = 'mongorestore'
    path = raw_input('Folder to recover from? ')
    dirpath = os.path.dirname(os.path.abspath(__file__))
    dirpath = dirpath[:-3]
    path = dirpath + path
    subprocess.call([cmd, path])
    client = MongoClient()
    db = client.db
    return db
