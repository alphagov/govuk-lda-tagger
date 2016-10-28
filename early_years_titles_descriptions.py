import csv
import json
import sys
from gensim_engine import GensimEngine

# Allow engine arguments to be passed on the command line
args = sys.argv[1:] if __name__ == '__main__' else ()

raw_documents = []

print("Reading input file 'input/early-years-titles-descriptions.csv'")
with open('input/early-years-titles-descriptions.csv', 'r') as f:
    reader = csv.reader(f)
    raw_documents = list(reader)

print("Prepare documents")
documents = [{'base_path': doc[0], 'text': doc[1]} for doc in raw_documents if len(doc) == 2 and doc[1] != '']

engine = GensimEngine(documents, log=True, dictionary_path='input/dictionary.txt', args=args)
engine.train(number_of_topics=5, passes=200)

print("Print topics to file")
topics_file = open('output/early_years_title_description_topics.csv', 'w')
for topic in engine.topics:
    topic_string = "{},{}\n".format(topic['topic_id'], topic['words'])
    topics_file.write(topic_string)
topics_file.close()

print("Prepare documents")
untagged_documents = [{'base_path': doc[0], 'text': doc[1]} for doc in raw_documents]

print("Tagging documents")
tagged_documents = engine.tag(untagged_documents)

print('Print tagged documents to file')
tagged_documents_file = open('output/early_years_title_description_tagged_data.csv', 'w')
for tagged_document in tagged_documents:
    tag_string = '{},{}\n'.format(tagged_document['base_path'], tagged_document['tags'])
    tagged_documents_file.write(tag_string)
tagged_documents_file.close()
