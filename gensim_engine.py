import argparse
import csv
import logging
import re
import sys
from operator import itemgetter
from gensim import corpora, models
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases
from nltk.corpus import stopwords
from collections import Counter
import pyLDAvis
import pyLDAvis.gensim

import gensim

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--nobigrams', dest='bigrams', action='store_false')

class GensimEngine:
    def __init__(self, documents, log=False, dictionary_path=None):
        """
        Documents is expected to be a list of dictionaries, where each element
        includes a `base_path` and `text`.
        """
        self.documents = documents
        self.lemmas = []
        self.dictionary = []
        self.topics = []
        self.corpus = []
        self.ldamodel = None
        self.bigrams = []
        self.top_bigrams = []
        self.dictionary_path = dictionary_path
        self.options = parser.parse_args()

        with open('input/bigrams.csv', 'r') as f:
            reader = csv.reader(f)
            self.top_bigrams = [bigram[0] for bigram in list(reader)]

        if log:
            logging.basicConfig(
                format='%(asctime)s : %(levelname)s : %(message)s',
                level=logging.INFO)


    def fetch_document_bigrams(self, document_lemmas, number_of_bigrams=100):
        """
        Given a number of lemmas identifying a document, it calculates N bigrams
        found in that document, where N=number_of_bigrams.
        """
        if not self.options.bigrams:
            return []

        bigram = Phrases()
        bigram.add_vocab([document_lemmas])
        bigram_counter = Counter()

        for key in bigram.vocab.keys():
            if key not in stopwords.words("english"):
                if len(key.split("_")) > 1:
                    bigram_counter[key] += bigram.vocab[key]

        found_bigrams = [[bigram] * bigram_count for bigram, bigram_count in bigram_counter.most_common(number_of_bigrams)]

        return found_bigrams


    def train(self, number_of_topics=20, words_per_topic=8, passes=50):
        """
        It trains the TF-IDF algorithm against the documents set in the
        initializer. We can control the number of topics we need and how many
        iterations the algorithm should make.
        """
        print("Generating lemmas for each of the documents")
        for document in self.documents:
            raw_text = document['text'].lower()
            all_lemmas = lemmatize(raw_text, allowed_tags=re.compile('(NN|JJ)'), stopwords=STOPWORDS)
            document_bigrams = self.fetch_document_bigrams(all_lemmas)
            known_bigrams = [bigram for bigram in document_bigrams if bigram in self.top_bigrams]
            self.lemmas.append(all_lemmas + known_bigrams)

        if self.dictionary_path:
            print("Load pre-existing dictionary from file")
            self.dictionary = corpora.Dictionary.load_from_text(self.dictionary_path)
        else:
            print("Turn our tokenized documents into a id <-> term dictionary")
            self.dictionary = corpora.Dictionary(self.lemmas)

        print("Convert tokenized documents into a document-term matrix")
        self.corpus = [self.dictionary.doc2bow(lemma) for lemma in self.lemmas]

        # print('Generating TF-IDF model')
        # tfidfmodel = gensim.models.TfidfModel(self.corpus)
        # corpus_tfidf = tfidfmodel[self.corpus]

        print("Generate TF-IDF model")
        self.ldamodel = gensim.models.ldamodel.LdaModel(
            # corpus_tfidf,
            self.corpus,
            num_topics=number_of_topics,
            id2word=self.dictionary,
            passes=passes)

        print('Saving topic information')
        raw_topics = self.ldamodel.show_topics(
            num_topics=number_of_topics,
            num_words=words_per_topic,
            formatted=False)

        self.topics = [{'topic_id': topic_id, 'words': words} for topic_id, words in raw_topics]


    def tag(self, untagged_documents, top_topics=3):
        """
        Given a list of documents (dictionary with `base_path` and `text`), this
        method adds an extra key to each of the dictionaries with the top 3
        topics associated with the document.
        """
        tagged_documents = []
        for document in untagged_documents:
            tagged_documents.append(self.topics_for(document, top_topics))

        return tagged_documents


    def topics_for(self, document, top_topics=3):
        """
        Given a single document, it returns a list of topics for the document.
        """
        raw_text = document['text'].lower()

        # Extract the lemmas from the raw text
        document_lemmas = lemmatize(raw_text, stopwords=STOPWORDS)

        # Extract the most common bigrams in the document
        document_bigrams = self.fetch_document_bigrams(document_lemmas)
        known_bigrams = [bigram for bigram in document_bigrams if bigram in self.top_bigrams]

        # Calculate the bag of words
        document_bow = self.dictionary.doc2bow(document_lemmas + known_bigrams)

        # Tag the document
        all_tags = self.ldamodel[document_bow]
        tags = sorted(all_tags, key=itemgetter(1), reverse=True)[:top_topics]
        document['tags'] = tags

        return document
        
        
    def visualise(self):
        """
        Visualise the topics generated 
        """
        
        # Create visualisation
        viz = pyLDAvis.gensim.prepare(self.ldamodel, self.corpus, self.dictionary)   
        
        # Output HTML object
        pyLDAvis.save_html(data=viz, fileobj='output/viz.html')
        
        print "Saving to viz.htm"


