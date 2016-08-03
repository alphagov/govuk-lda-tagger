import logging
from operator import itemgetter
from gensim import corpora, models
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
import gensim

class GensimEngine:
    def __init__(self, documents, log=False):
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

        if log:
            logging.basicConfig(
                format='%(asctime)s : %(levelname)s : %(message)s',
                level=logging.INFO)


    def train(self, number_of_topics=20, words_per_topic=10, passes=50):
        """
        It trains the LDA algorithm against the documents set in the
        initializer. We can control the number of topics we need and how many
        iterations the algorithm should make.
        """
        print("Generating lemmas for each of the documents")
        for document in self.documents:
            raw_text = document['text'].lower()
            tokens = lemmatize(raw_text, stopwords=STOPWORDS)
            self.lemmas.append(tokens)

        print("Turn our tokenized documents into a id <-> term dictionary")
        self.dictionary = corpora.Dictionary(self.lemmas)

        print("Convert tokenized documents into a document-term matrix")
        self.corpus = [self.dictionary.doc2bow(lemma) for lemma in self.lemmas]

        print("Generate LDA model")
        self.ldamodel = gensim.models.ldamodel.LdaModel(
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
        document_lemmas = lemmatize(raw_text, stopwords=STOPWORDS)
        document_bow = self.dictionary.doc2bow(document_lemmas)
        all_tags = self.ldamodel[document_bow]
        tags = sorted(all_tags, key=itemgetter(1), reverse=True)[:top_topics]
        document['tags'] = tags

        return document
