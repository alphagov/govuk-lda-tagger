import nltk
import csv
import ipdb
import string
from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from collections import Counter

print("Reading input file 'input/audits_with_content.csv'")
with open('input/audits_with_content.csv', 'r') as f:
    reader = csv.reader(f)
    raw_documents = list(reader)

print("Prepare documents")
documents = [doc[2] for doc in raw_documents if doc[2] != '']
sentences = []
bigram = Phrases()

for document in documents:
    raw_text = document.lower()
    tokens = lemmatize(raw_text, stopwords=STOPWORDS)
    sentences.append(tokens)
    bigram.add_vocab([tokens])

bigram_counter = Counter()
for key in bigram.vocab.keys():
    if key not in stopwords.words("english"):
        if len(key.split("_")) > 1:
            bigram_counter[key] += bigram.vocab[key]

for key, counts in bigram_counter.most_common(200):
    print '{0: <20} {1}'.format(key.encode("utf-8"), counts)
