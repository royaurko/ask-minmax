from __future__ import print_function
import subprocess
import helper
from pymongo import MongoClient


def connect():
    """
    Get the host and port number from the user and connect a client
    :return: Connected Mongoclient
    """
    host = raw_input('Mongodb host name (press enter for localhost): ')
    while True:
        try:
            port = raw_input('Port number (press enter for default port): ')
            if not port:
                break
            port = int(port)
            break
        except ValueError:
            helper.error_number()
    if not host and not port:
        client = MongoClient()
    elif not host:
        client = MongoClient(port=port)
    elif not port:
        client = MongoClient(host=host)
    else:
        client = MongoClient(host=host, port=port)
    return client


def initialize_db():
    """ Connect client, drop all tables in the database and return
    :return: client, database db with all tables dropped
    """
    client = connect()
    db = client.db
    db.problems.drop()
    db.questions.drop()
    return (client, db)


def dump_db():
    """ Dump database to BSON file
    :return: None
    """
    try:
        path = raw_input('Name of folder to dump BSON files to (default = database)? ')
        cmd = 'mongodump'
        if not path:
            path = 'database'
        output = subprocess.call([cmd, '-o', path, '--db', 'db'])
        print(output)
    except:
        print('Error backing up database!')


def recover_db(client):
    """ Recover database from BSON file
    :param client:
    :return:
    """
    client.drop_database('db')
    cmd = 'mongorestore'
    path = raw_input('Folder to recover from (default = database)? ')
    if not path:
        path = 'database'
    subprocess.call([cmd, path])
    client = MongoClient()
    db = client.db
    return db
