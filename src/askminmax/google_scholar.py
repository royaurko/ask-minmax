from __future__ import print_function
from scholar import ScholarQuerier, ScholarSettings, SearchScholarQuery, csv
import requests
from dragnet import content_extractor
import sys
import hashlib
import time


def download_abstracts_scholar(db, num_results, keyword):
    """ Download abstracts from google scholar
    :param db:
    :param num_results:
    :param keyword:
    :return:
    """
    querier = ScholarQuerier()
    settings = ScholarSettings()
    querier.apply_settings(settings)
    query = SearchScholarQuery()
    query.set_phrase(keyword)
    query.set_num_page_results(min(20, num_results))
    total = 0
    while total < num_results:
        query.set_start(total)
        querier.send_query(query)
        querier.save_cookies()
        items = csv(querier)
        for index, item in enumerate(items):
            url = item.strip().split('|')[1]
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
            if item is None:
                        d = {'keyword': keyword, 'abstract': content, 'text': text, 'hash': hash_value}
                        db.papers.insert(d)
        time.sleep(300)
        total += 20