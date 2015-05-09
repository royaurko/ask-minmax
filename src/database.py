# Contains all the functions responsible for maintaining the database

import subprocess
import os
import helper
from pymongo import MongoClient


def connect():
    host = raw_input('Mongodb host name (press enter for localhost): ')
    while True:
        try:
            port = raw_input('Port number (press enter for default port): ')
            if not port:
                break
            port = int(port)
            break
        except ValueError:
            helper.errornumber()
    if not host and not port:
        client = MongoClient()
    elif not host:
        client = MongoClient(port=port)
    elif not port:
        client = MongoClient(host=host)
    else:
        client = MongoClient(host=host, port=port)
    return client


def initializedb():
    '''Initialize the database with problems and questions'''
    client = connect()
    db = client.db
    db.problems.drop()
    db.questions.drop()
    return (client, db)


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


def recoverdb(client):
    '''Recover a db from a bson dump'''
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
