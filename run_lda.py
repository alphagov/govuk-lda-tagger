from __future__ import print_function
import csv
import ipdb
import os
import lda
import numpy as np

import lda.utils as lda_utils

from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS

print("Reading input file 'input/audits_with_content.csv'")
with open('input/audits_with_content.csv', 'r') as f:
    reader = csv.reader(f)
    documents = list(reader)

# Remove documents without body
documents = [doc for doc in documents if doc[2] != '']
doc_count = len(documents)

titles = []
texts = []
documents_words = []

print("Generating titles for all documents")
for index, document in enumerate(documents):
    link = document[0]
    slug = link.split('/')[-1]
    title = slug.replace('-', ' ')
    titles.append({'id': index, 'title': title})

print("Writing titles into output file")
with open('output/data.titles', 'w') as csvfile:
    fieldnames = ['id', 'title']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for title_record in titles:
        writer.writerow(title_record)

print("Generating tokens for every document")
for index, document in enumerate(documents):
    print(str(index), " of ", str(doc_count), end='\r')
    raw_text = document[2].lower()
    tokens = lemmatize(raw_text, stopwords=STOPWORDS)

    documents_words.append(tokens)
    for token in tokens:
        texts.append(token)

tokens_set = set(texts)

print("Writing tokens into output file")
with open('output/data.tokens', 'w') as f:
    for token in tokens_set:
        print(token, file=f)

print("Generating Document-Term Matrix (DTM)")
dtm = np.empty((len(documents_words), len(tokens_set)), dtype=np.intc)
tokens_count = len(tokens_set)

for token_index, token in enumerate(tokens_set):
    print(str(token_index), " of ", str(tokens_count), end='\r')
    for doc_index, document in enumerate(documents_words):
        dtm[doc_index, token_index] = document.count(token)

print("Generating LDAC data")
doclines = list(lda_utils.dtm2ldac(dtm))

print("Writing LDAC file")
with open('output/data.ldac', 'w') as f:
    for line in doclines:
        print(line, file=f)

def load_govuk_data():
    ldac_fn = os.path.join('output', 'data.ldac')
    return lda.utils.ldac2dtm(open(ldac_fn), offset=0)


def load_govuk_tokens():
    tokens_fn = os.path.join('output', 'data.tokens')
    with open(tokens_fn) as f:
        tokens = tuple(f.read().split())
    return tokens


def load_govuk_titles():
    govuk_titles_fn = os.path.join('output', 'data.titles')
    with open(govuk_titles_fn) as f:
        titles = tuple(line.strip() for line in f.readlines())
    return titles

print("Loading LDAC")
X = load_govuk_data()
print("Loading tokens")
tokens = load_govuk_tokens()
print("Loading titles")
titles = load_govuk_titles()

model = lda.LDA(n_topics=20, n_iter=100, random_state=1)
model.fit(X)
topic_word = model.topic_word_
n_top_words = 5

print("Derived topics from documents:")
for i, topic_dist in enumerate(topic_word):
     topic_words = np.array(tokens)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_

ipdb.set_trace()

f = open('output/data.txt', 'w')
result = [ "{} (top topic: {})".format(t, dt.argmax()) for t, dt in zip(titles,
    doc_topic)]
f.write("\n".join(result))
f.close()
