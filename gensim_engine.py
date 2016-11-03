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
from corpus_building import CorpusReader
import pyLDAvis
import pyLDAvis.gensim

import gensim
warnings.filterwarnings('error')

csv.field_size_limit(sys.maxsize)


class GensimEngine(object):
    def __init__(self, corpus, dictionary, log=False, corpus_reader=None):
        """
        The corpus is a bag of words (a list of lists of tuples)
        The dictionary maps word ids to strings
        """
        self.topics = []
        self.ldamodel = None
        self.corpus = corpus
        self.dictionary = dictionary

        # TODO: this shouldn't be needed
        self.corpus_reader = corpus_reader

        if log:
            logging.basicConfig(
                format='%(asctime)s : %(levelname)s : %(message)s',
                level=logging.INFO)

    @staticmethod
    def from_documents(documents, log=False, dictionary_path=None, include_bigrams=True, use_phrasemachine=False):
        """
        Documents is expected to be a list of dictionaries, where each element
        includes a `base_path` and `text`.
        """
        reader = CorpusReader(include_bigrams=include_bigrams, use_phrasemachine=use_phrasemachine)
        corpus, dictionary = reader.build_corpus(documents, dictionary_path=dictionary_path)
        return GensimEngine(corpus, dictionary, log=log, corpus_reader=reader)

    @staticmethod
    def from_experiment(name, log=False):
        experiment = Experiment.load(name)
        return GensimEngine(experiment.corpus, experiment.dictionary, log=log)

    def train(self, number_of_topics=20, words_per_topic=8, passes=50):
        """
        It trains the TF-IDF algorithm against the documents set in the
        initializer. We can control the number of topics we need and how many
        iterations the algorithm should make.
        """
        print("Generate TF-IDF model")
        self.ldamodel = gensim.models.ldamodel.LdaModel(
            # corpus_tfidf,
            self.corpus,
            num_topics=number_of_topics,
            id2word=self.dictionary,
            passes=passes
        )

        raw_topics = self.ldamodel.show_topics(
            num_topics=number_of_topics,
            num_words=words_per_topic,
            formatted=False
        )

        self.topics = [{'topic_id': topic_id, 'words': words} for topic_id, words in raw_topics]

        return Experiment(model=self.ldamodel, corpus=self.corpus, dictionary=self.dictionary)

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
        if self.corpus_reader is None:
            # FIXME retrieve the document bow from the corpus
            raise NotImplementedError

        raw_text = document['text'].lower()

        document_phrases = self.corpus_reader.document_bow(raw_text)
        document_bow = self.dictionary.doc2bow(document_phrases)

        # Tag the document
        all_tags = self.ldamodel[document_bow]
        tags = sorted(all_tags, key=itemgetter(1), reverse=True)[:top_topics]
        document['tags'] = tags

        return document


class Experiment(object):
    """
    Each experiment contains a corpus of words and an LDA model of it
    """
    DEFAULT_EXPERIMENT_PATH = os.path.join('output', 'models')

    def __init__(self, model, corpus, dictionary):
        self.ldamodel = model
        self.corpus = corpus
        self.dictionary = dictionary

    @staticmethod
    def load(experiment_name, path=DEFAULT_EXPERIMENT_PATH):
        model_filename = Experiment._filename(path, experiment_name, 'model')
        corpus_filename = Experiment._filename(path, experiment_name, 'corpus')
        dictionary_filename = Experiment._filename(path, experiment_name, 'dict')

        model = gensim.models.ldamodel.LdaModel.load(model_filename)
        corpus = list(corpora.MmCorpus(corpus_filename))
        dictionary = corpora.Dictionary.load_from_text(dictionary_filename)

        return Experiment(model=model, corpus=corpus, dictionary=dictionary)

    def save(self, experiment_name, path=DEFAULT_EXPERIMENT_PATH):
        model_filename = self._filename(path, experiment_name, 'model')
        corpus_filename = self._filename(path, experiment_name, 'corpus')
        dictionary_filename = self._filename(path, experiment_name, 'dict')

        self.ldamodel.save(model_filename)
        corpora.MmCorpus.serialize(corpus_filename, self.corpus)
        self.dictionary.save_as_text(dictionary_filename)

    def visualise(self, filename):
        """
        Visualise the topics generated
        """
        # Create visualisation
        viz = pyLDAvis.gensim.prepare(self.ldamodel, self.corpus, self.dictionary)

        # Output HTML object
        pyLDAvis.save_html(data=viz, fileobj=filename)

        print "Saving to viz.htm"

    def topics(self, number_of_topics=20, words_per_topic=8):
        raw_topics = self.ldamodel.show_topics(
            num_topics=number_of_topics,
            num_words=words_per_topic,
            formatted=False)

        return [{'topic_id': topic_id, 'words': words} for topic_id, words in raw_topics]

    @staticmethod
    def _filename(path, experiment, suffix):
        return os.path.join(path, '{}_{}'.format(experiment, suffix))
