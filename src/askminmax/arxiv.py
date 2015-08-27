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


def download(db, keywords):
    ''' Download abstracts and fulltexts from arxiv
    :param db: The Mongodb database to download the papers to
    :param keywords: List of keywords to search for
    :return: None, modify database in place
    '''

    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?'
    feedparserurl = 'http://a9.com/-/spec/opensearch/1.1/'
    atomurl = 'http://arxiv.org/schemas/atom'
    feedparser._FeedParserMixin.namespaces[feedparserurl] = 'opensearch'
    feedparser._FeedParserMixin.namespaces[atomurl] = 'arxiv'
    # Downloads pdfs here, converts to text file here
    dirname = 'arxiv-data'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    pdfname = dirname + '/fullpaper.pdf'
    textname = dirname + '/fullpaper.txt'
    for keyword in keywords:
        search_query = 'all:' + keyword
        start = 0
        max_results = 1000
        while True:
            query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                                 start,
                                                                 max_results)
            response = urllib2.urlopen(base_url+query).read()
            # perform a GET request using the base_url and query
            response = urllib2.urlopen(base_url+query).read()
            # change author -> contributors (because contributors is a list)
            response = response.replace('author', 'contributor')
            # parse the response using feedparser
            feed = feedparser.parse(response)
            # Print some stats on the screen, user is getting nervous
            print 'Feed title: %s' % feed.feed.title
            print 'totalResults for this query: %s' % feed.feed.opensearch_totalresults
            print 'itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage
            print 'startIndex for this query: %s' % feed.feed.opensearch_startindex
            # Quit if total search results are 0
            if feed.feed.opensearch_totalresults <= 0:
                break
            for entry in feed.entries:
                print 'arxiv-id: %s' % entry.id.split('/abs/')[-1]
                # get the links to the abs page and pdf for this e-print
                for link in entry.links:
                    if link.rel == 'alternate':
                        print 'abs page link: %s' % link.href
                    elif link.title == 'pdf':
                        pdflink = link.href
                        pdflink = pdflink[:7] + 'lanl.' + pdflink[7:]
                        print 'pdf link: %s' % pdflink
                try:
                    comment = entry.arxiv_comment
                except AttributeError:
                    comment = 'No comment found'
                # Don't bother with withdrawn papers
                if 'withdraw' in comment.lower():
                    continue
                # Sleep for a minute otherwise you'll get banned
                time.sleep(60)

                # Download the pdf
                if pdflink:
                    pdfresponse = urllib2.urlopen(pdflink)
                    pdf = open(pdfname, 'wb')
                    pdf.write(pdfresponse.read())
                    pdf.close()
                    os.system("pdftotext -enc ASCII7 '%s' '%s'" %(pdfname, textname))

                try:
                    textfile = open(textname, 'r')
                    text = textfile.read().replace('\n', '')
                    # text = scrunch(text)
                except:
                    # Couldn't convert pdf to text for whatever reason, skip
                    continue

                # Add to database
                summary = entry.summary
                hashval = hashlib.md5(summary).hexdigest()

                item = db.papers.find_one({'hash': hashval})

                if item is None:
                    d = {'keyword': keyword, 'abstract': summary, 'text': text, 'hash': hashval}
                    db.papers.insert(d)
                    start += 1