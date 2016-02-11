from .scholar import ScholarQuerier, ScholarSettings, SearchScholarQuery, csv
import requests
from dragnet import content_extractor
import sys
import hashlib
import time
from . import database
import random
import argparse


def download_abstracts_scholar(db, start, num_results, keyword, time_delay=1200):
    """ Download abstracts from google scholar
    :param db: Mongodb database
    :param start: The start page
    :param num_results: Number of results
    :param keyword: Keyword to search for
    :param time_delay: time delay
    :return: None
    """
    querier = ScholarQuerier()
    settings = ScholarSettings()
    querier.apply_settings(settings)
    query = SearchScholarQuery()
    query.set_phrase(keyword)
    query.set_num_page_results(min(20, num_results))
    total = start
    while total < num_results:
        try:
            query.set_start(total)
            querier.send_query(query)
            # querier.save_cookies()
            items = csv(querier)
            for index, item in enumerate(items):
                url = item.strip().split('|')[1]
                content = ''
                try:
                    r = requests.get(url)
                    try:
                        content = content_extractor.analyze(r.content)
                    except Exception as e:
                        sys.stderr.write('Error fetching content: ' + str(e) + '\n')
                except requests.packages.urllib3.exceptions.ProtocolError:
                        sys.stderr.write('Error: ' + str(e) + '\n')
                except requests.exceptions.RequestException as e:
                        sys.stderr.write('Error fetching URL ' + url + ': ' + str(e) + '\n')
                except Exception as e:
                        sys.stderr.write('Error fetching URL ' + url + ': ' + str(e) + '\n')
                print(" --------- Abstract %d  ------------ " % (index + 1 + total))
                print(content)
                hash_value = hashlib.md5(content).hexdigest()
                item = db.papers.find_one({'hash': hash_value})
                text = str()
                acm_msg = 'Did you know the ACM DL App is now available? Did you know your Organization can subscribe to the ACM Digital Library?'
                if item is None and content != acm_msg:
                            d = {'keyword': keyword, 'abstract': content, 'text': text, 'hash': hash_value}
                            db.papers.insert(d)
            delay = random.randint(time_delay, time_delay + 600)
            print('Sleeping for %d seconds ... ' % delay)
            time.sleep(delay)
            total += 20
        except KeyboardInterrupt:
            break
    database.dump_db()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--db', help='Database to store result in')
    parser.add_argument('-s', '--start', help='Start number of abstract to download')
    parser.add_argument('-n', '--num_results', help='Number of abstracts to download')
    parser.add_argument('-k', '--keyword', help='Keyword to download')
    parser.add_argument('-t', '--time', help='Time delay')
    args = parser.parse_args()
    download_abstracts_scholar(args.db, args.start, args.num_results, args.keyword, args.time)
