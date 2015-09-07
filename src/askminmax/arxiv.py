from __future__ import print_function
import urllib2
import feedparser
import os
import hashlib
import re
import time


# A definition of keywords that might be relevant, you can use your own to download
k = "TSP, matching, network flow, approximation algorithm, routing, combinatorial optimization"


def clean(to_translate):
    ''' Clean text
    :param to_translate: Text to clean
    :return: Cleaned text
    '''
    translate_to = u' '
    not_letters_or_digits = u'$!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char), translate_to) for
                           char in not_letters_or_digits)
    return to_translate.translate(translate_table)


def download(db, flag, keywords):
    ''' Download abstracts and fulltexts from arxiv
    :param db: The Mongodb database to download the papers to
    :param flag: flag in {Yes, No}, if Yes downloads pdfs
    :param keywords: List of keywords to search for
    :return: None, modify database in place
    '''

    # Base API query url
    base_url = 'http://export.arxiv.org/api/query?'
    feedparser_url = 'http://a9.com/-/spec/opensearch/1.1/'
    atom_url = 'http://arxiv.org/schemas/atom'
    feedparser._FeedParserMixin.namespaces[feedparser_url] = 'opensearch'
    feedparser._FeedParserMixin.namespaces[atom_url] = 'arxiv'
    # Downloads pdfs here, converts to text file here
    dir_name = 'arxiv-data'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    pdf_name = dir_name + '/fullpaper.pdf'
    text_name = dir_name + '/fullpaper.txt'
    for keyword in keywords:
        search_query = 'all:' + keyword
        start = 0
        max_results = 10000
        query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                             start, max_results)
        print(base_url + query)
        response = urllib2.urlopen(base_url + query).read()
        # change author -> contributors (because contributors is a list)
        response = response.replace('author', 'contributor')
        # parse the response using feedparser
        feed = feedparser.parse(response)
        # Print some stats on the screen, user is getting nervous
        print('Feed title: %s' % feed.feed.title)
        print('totalResults for this query: %s' % feed.feed.opensearch_totalresults)
        print('itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage)
        print('startIndex for this query: %s' % feed.feed.opensearch_startindex)
        # Continue if total search results are 0
        if feed.feed.opensearch_totalresults <= 0:
            continue
        for entry in feed.entries:
            print('arxiv-id: %s' % entry.id.split('/abs/')[-1])
            # get the links to the abs page and pdf for this e-print
            for link in entry.links:
                if link.rel == 'alternate':
                    continue
                elif link.title == 'pdf':
                    pdf_link = link.href
                    pdf_link = pdf_link[:7] + 'lanl.' + pdf_link[7:]
            try:
                comment = entry.arxiv_comment
            except AttributeError:
                comment = 'No comment found'
            # Don't bother with withdrawn papers
            if 'withdraw' in comment.lower():
                continue
            text = ''
            if flag:
                # Download the pdf
                # Sleep for a minute otherwise you'll get banned
                time.sleep(60)
                if pdf_link:
                    pdf_response = urllib2.urlopen(pdf_link)
                    pdf = open(pdf_name, 'wb')
                    pdf.write(pdf_response.read())
                    pdf.close()
                    os.system("pdftotext -enc ASCII7 '%s' '%s'" %(pdf_name, text_name))
                try:
                    text_file = open(text_name, 'r')
                    text = text_file.read().replace('\n', '')
                    # text = scrunch(text)
                except:
                    # Couldn't convert pdf to text for whatever reason, skip
                    continue
            # Add to database
            summary = entry.summary
            hash_value = hashlib.md5(summary).hexdigest()
            item = db.papers.find_one({'hash': hash_value})
            if item is None:
                d = {'keyword': keyword, 'abstract': summary, 'text': text, 'hash': hash_value}
                db.papers.insert(d)
