from __future__ import print_function
import numpy as np
import scipy.stats as stats
from gensim import matutils


def relative_entropy(p, q):
    """
    A measure of the difference between two probability distributions P and Q
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    return np.sum([stats.entropy(p, q), stats.entropy(q, p)])


class ModelEvaluator(object):
    """
    Evaluate LDA models using Symmetric KL divergence.
    """
    def __init__(self, corpus):
        self.corpus = corpus
        self.document_term_counts = np.array([sum(cnt for _, cnt in doc) for doc in corpus])


    def __call__(self, model):
        """
        Evaluate the model. Higher is better.
        """
        # numpy.exp(dirichlet_expectation(model.state.sstats))
        m1 = model.expElogbeta

        # Singular Value Decomposition
        U, cm1, V = np.linalg.svd(m1)

        topics = model[self.corpus]
        document_term_matrix = matutils.corpus2dense(topics, model.num_topics).transpose()

        cm2 = self.document_term_counts.dot(document_term_matrix)
        cm2 += 0.0001

        cm2norm = np.linalg.norm(self.document_term_counts)
        cm2 = cm2/cm2norm

        return relative_entropy(cm1, cm2)


if __name__ == '__main__':
    # Test the evaluator comes up with a metric for each possible model
    import gensim_engine
    import model_io
    import sys
    documents = model_io.load_documents(sys.argv[1])
    engine = gensim_engine.GensimEngine(documents)
    evaluator = ModelEvaluator(engine.corpus)

    for number_of_topics, _model, metric in engine._evaluate_number_of_topics(evaluator, min_topics=6, max_topics=21):
        print('{:-2d} topics: {:.2f}'.format(number_of_topics, metric))
