# Organization of the code

The main files that comprise this expert system are:

* [helper.py](helper.py)
* [database.py](database.py)
* [problems.py](problems.py)
* [questions.py](questions.py)
* [sepquestions.py](sepquestions.py)
* [training.py](training.py)
* [expert.py](expert.py)

Below we list the methods with their parameters and return types with a one line description of what they do.

## [helper.py](helper.py)

* Depends on: `hashlib`, `re`
* Functions:
    - `None erroronzero()`: Error message
    - `None erroronnumber():` Error message
    - `s strip(s):` Strip leading spaces from `s` from left and right
    - `val gethashval(s):` Get hash vaulue of `s` after removing non-alpha chars from it
    - `m mass(db, table, hashval, property)`: Look up item with hashval from db.table and return mass of property
    
    
## [database.py](database.py)

* Depends onL `subprocess`, `helper`, `pymongo`
* Functions:
    - `client connect()`: Get the host and port number from the user and connect a client
    - `(client, db) initializedb()`: Connect client, drop all tables in the database and return
    - `None dumpdb(db)`: Dump database to BSON file
    - `db recoverdb(client)`: Recover db from BSON file
    
## [problems.py](problems.py)


## [questions.py](questions.py)


## [sepquestions.py](sepquestions.py)

## [training.py](training.py)

## [expert.py](expert.py)

