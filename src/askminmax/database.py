import subprocess
import helper
from pymongo import MongoClient


def connect():
    '''
    Get the host and port number from the user and connect a client
    :return: Connected Mongoclient
    '''
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
    ''' Connect client, drop all tables in the database and return
    :return: client, database db with all tables dropped
    '''
    client = connect()
    db = client.db
    db.problems.drop()
    db.questions.drop()
    return (client, db)


def dumpdb(db):
    ''' Dump database to BSON file
    :param db: Name of the database to dump to BSON files
    :return: None
    '''
    try:
        path = raw_input('Name of folder to dump BSON files to (default = db)? ')
        cmd = 'mongodump'
        output = subprocess.call([cmd, '-o', path])
        print output
    except:
        print 'Error backing up database!'


def recoverdb(client):
    ''' Recover database from BSON file
    :param client:
    :return:
    '''
    client.drop_database('db')
    cmd = 'mongorestore'
    path = raw_input('Folder to recover from (default = db)? ')
    subprocess.call([cmd, path])
    client = MongoClient()
    db = client.db
    return db