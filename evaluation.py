from __future__ import print_function
import numpy as np
import scipy.stats as stats
from gensim import matutils


def relative_entropy(p, q):
    """
    A measure of the difference between two probability distributions P and Q
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    This is a measure of the amount of information that is lost when predicting
    probability distribution P based on a model Q.  p(P||Q) =/= p(Q||P)
    i.e. non-symmetric, so calculate entropy both ways, and sum to calculate.
    """
    return np.sum([stats.entropy(p, q), stats.entropy(q, p)])


class ModelEvaluator(object):
    """
    Evaluate LDA models using Symmetric Kullback-LeiblerL divergence.
    This implementation is explained here https://archive.is/KBGwt, 
    Gist: https://gist.github.com/cigrainger/62910e58db46b7397de2
    Based on Arun et al: http://link.springer.com/chapter/10.1007/978-3-642-13657-3_43
    """
    def __init__(self, corpus):
        self.corpus = corpus
        self.document_term_counts = np.array([sum(cnt for _, cnt in doc) for doc in corpus])


    def __call__(self, model):
        """
        Evaluate the model. Higher is better.
        """
        # numpy.exp(dirichlet_expectation(model.state.sstats))
        # .expElogbeta gives P(word|topic)
        m1 = model.expElogbeta

        # Singular Value Decomposition to create the sigma matrix
        # A Diagonal m x n matrix 
        U, cm1, V = np.linalg.svd(m1)

        topics = model[self.corpus]
        # corpus2dense converts sparse matrix to dense
        document_term_matrix = matutils.corpus2dense(topics, model.num_topics).transpose()
        
        cm2 = self.document_term_counts.dot(document_term_matrix)
        
        # Adding negligible values to cm2 is required on very
        # corpuses, so we can ignore here
        #cm2 += 0.0001

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
