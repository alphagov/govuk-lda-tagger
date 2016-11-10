import glob
import argparse
import csv
import logging
import re
import sys
import warnings
import os
from itertools import chain, repeat
from operator import itemgetter
from gensim import corpora, models
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Phrases
from nltk.corpus import stopwords
from collections import Counter
import pyLDAvis
import pyLDAvis.gensim
import phrasemachine
import textacy
from textacy import preprocess

import gensim

PUNCTUATION_REGEX = re.compile(r'\W|[_0-9]', flags=re.UNICODE)

def preprocess_unicode(raw_text):
    raw_text = preprocess.transliterate_unicode(raw_text.lower())
    raw_text = preprocess.replace_urls(raw_text, replace_with=u'')
    raw_text = preprocess.replace_emails(raw_text, replace_with=u'')
    raw_text = preprocess.replace_phone_numbers(raw_text, replace_with=u'')
    raw_text = preprocess.replace_numbers(raw_text, replace_with=u'')
    raw_text = preprocess.replace_currency_symbols(raw_text, replace_with=u'')
    return raw_text

def load_stopwords():
    stopwords = []
    for filename in glob.glob('stopwords/*.txt'):
        with open(filename) as fileobj:
            for line in fileobj:
                line = preprocess_unicode(line.decode('utf8').strip())
                line = preprocess.remove_punct(line)
                if line:
                    stopwords.append(line)

    return stopwords + [word.decode('utf8') for word in STOPWORDS]


STOPWORDS_UNICODE = load_stopwords()
STOPWORDS_BYTES = [word.encode('utf8') for word in STOPWORDS_UNICODE]


class CorpusReader(object):
    """
    Extract terms and phrases from raw text to run LDA on.
    """
    def __init__(self, include_bigrams=True, use_phrasemachine=False, use_textacy=False, use_lemmatisation=False, use_tfidf=False, no_below=20, no_above=0.15, keep_n=None):
        self.include_bigrams = include_bigrams

        self.use_phrasemachine = use_phrasemachine
        self.use_textacy = use_textacy
        self.use_lemmatisation = use_lemmatisation
        self.use_tfidf = use_tfidf
        self.no_below = no_below
        self.no_above = no_above
        self.keep_n = keep_n

        with open('input/bigrams.csv', 'r') as f:
            reader = csv.reader(f)
            self.top_bigrams = [bigram[0] for bigram in list(reader)]

    def document_phrases(self, raw_text):
        """
        Extract some kind of n-grams from a document
        """
        phrases = []

        if self.use_phrasemachine:
            phrases += self._phrases_in_raw_text_via_phrasemachine(raw_text)

        if self.use_lemmatisation:
            phrases += self._phrases_in_raw_text_via_lemmatisation(raw_text)

        if self.use_textacy:
            phrases += self._phrases_in_raw_text_via_textacy(raw_text)

        return phrases

    def fetch_document_bigrams(self, document_lemmas, number_of_bigrams=100):
        """
        Given a number of lemmas identifying a document, it calculates N bigrams
        found in that document, where N=number_of_bigrams.
        """
        if not self.include_bigrams:
            return []

        bigram = Phrases()
        bigram.add_vocab([document_lemmas])
        bigram_counter = Counter()

        for key in bigram.vocab.keys():
            if key not in STOPWORDS_BYTES:
                if len(key.split("_")) > 1:
                    bigram_counter[key] += bigram.vocab[key]

        bigram_iterators = [
            repeat(bigram, bigram_count)
            for bigram, bigram_count
            in bigram_counter.most_common(number_of_bigrams)
        ]
        found_bigrams = list(chain(*bigram_iterators))
        known_bigrams = [bigram for bigram in found_bigrams if bigram in self.top_bigrams]

        return known_bigrams

    def build_corpus(self, documents, dictionary_path=None):
        """
        Build a corpus and dictionary from the input documents.

        You can load an existing dictionary file to avoid computing it
        from scratch when retraining the model on the same input.
        """
        print("Generating lemmas for each of the documents")
        phrases = []
        for index, document in enumerate(documents):
            print("[{}] processing {}".format(str(index), document['base_path']))
            raw_text = preprocess_unicode(document['text'].decode('utf8'))

            phrases.append(self.document_phrases(raw_text))

        if dictionary_path:
            print("Load pre-existing dictionary from file")
            dictionary = corpora.Dictionary.load_from_text(dictionary_path)
        else:
            print("Turn our tokenized documents into a id <-> term dictionary")
            dictionary = corpora.Dictionary(phrases)

            # Filter out very (in)frequent words. This changes the id <-> term mapping.
            dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above, keep_n=self.keep_n)

        print("Convert tokenized documents into a document-term matrix")
        corpus = [dictionary.doc2bow(phrase) for phrase in phrases]

        if self.use_tfidf:
            print("Generate TF-IDF corpus")
            tfidfmodel = gensim.models.TfidfModel(corpus)
            corpus = tfidfmodel[corpus]


        return corpus, dictionary

    def _phrases_in_raw_text_via_textacy(self, raw_text):
        """
        Builds a list of phrases from raw text using textacy.
        """
        all_lemmas = lemmatize(raw_text, stopwords=STOPWORDS_UNICODE)
        curated_words = [word.split('/')[0] for word in all_lemmas]
        curated_text = ' '.join(curated_words)

        doc = textacy.Doc(unicode(curated_text.decode('ascii', 'ignore')), lang=u'en')

        all_phrases = []
        all_phrases += textacy.extract.ngrams(doc, 2, filter_stops=True, filter_punct=True, filter_nums=True)
        all_phrases += textacy.extract.ngrams(doc, 3, filter_stops=True, filter_punct=True, filter_nums=True)
        all_phrases += textacy.extract.ngrams(doc, 4, filter_stops=True, filter_punct=True, filter_nums=True)
        all_phrases += textacy.extract.ngrams(doc, 5, filter_stops=True, filter_punct=True, filter_nums=True)

        phrases = [unicode(phrase) for phrase in all_phrases]
        print(phrases)

        return phrases

    def _phrases_in_raw_text_via_phrasemachine(self, raw_text):
        """
        Builds a list of phrases from raw text using phrasemachine.
        """
        # This returns a Dictionary of counts
        phrase_counts = phrasemachine.get_phrases(raw_text)['counts']

        phrases_in_document = []
        for unique_phrase in phrase_counts:
            # Fetch how many times this phrase occurred
            phrase_count = phrase_counts[unique_phrase]

            # Create N strings based on the count, since LDA will do the counts
            phrases_for_phrase_count = [unique_phrase] * phrase_count

            # Now that we have the phrase repeated, add them to the final
            # list of phrases.
            for phrase in phrases_for_phrase_count:
                phrases_in_document.append(phrase)

        return phrases_in_document

    def _phrases_in_raw_text_via_lemmatisation(self, raw_text):
        """
        Builds a list of lemmas from raw text using lemmatization.
        """
        all_lemmas = lemmatize(raw_text, allowed_tags=re.compile('(NN|JJ)'), stopwords=STOPWORDS_UNICODE)
        document_bigrams = self.fetch_document_bigrams(all_lemmas)
        known_bigrams = [bigram for bigram in document_bigrams if bigram in self.top_bigrams]

        return (all_lemmas + known_bigrams)
