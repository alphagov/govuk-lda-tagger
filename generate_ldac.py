import csv
import ipdb
import numpy as np
import lda.utils as lda_utils
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

print("Reading input file 'input/audits_with_content.csv'")
with open('input/audits_with_content.csv', 'r') as f:
    reader = csv.reader(f)
    documents = list(reader)

documents = [doc for doc in documents if doc[2] != '']

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()

texts = []
documents_words = []

print("Grabbing relevant words from documents")
for index, document in enumerate(documents):
    raw_text = document[2].lower()
    tokens = tokenizer.tokenize(raw_text)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    documents_words.append(stemmed_tokens)

    for token in stemmed_tokens:
        texts.append(token)

tokens_set = set(texts)

print("Generating Document-Term Matrix (DMT)")
dtm = np.empty((len(documents_words), len(tokens_set)), dtype=np.intc)
tokens_count = len(tokens_set)

for token_index, token in enumerate(tokens_set):
    print(str(token_index), " of ", str(tokens_count), end='\r')
    for doc_index, document in enumerate(documents_words):
        dtm[doc_index, token_index] = document.count(token)

doclines = list(lda_utils.dtm2ldac(dtm))

print("Writing LDAC file")
with open('output/data.ldac', 'w') as f:
    for line in doclines:
        print(line, file=f)
