import os
import lda
import ipdb
import numpy as np

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

model = lda.LDA(n_topics=20, n_iter=1000, random_state=1)
model.fit(X)
topic_word = model.topic_word_
n_top_words = 5

print("Derived topics from documents:")
for i, topic_dist in enumerate(topic_word):
     topic_words = np.array(tokens)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_

f = open('output/data.txt', 'w')
[print("{} (top topic: {})".format(title, doc_top.argmax()), file=f) for topic, doc_top in zip(topics, doc_topic)]
