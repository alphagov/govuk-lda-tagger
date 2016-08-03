import csv
import json
from gensim_engine import GensimEngine

raw_documents = []

print("Reading input file 'input/audits_with_content.csv'")
with open('input/audits_with_content.csv', 'r') as f:
    reader = csv.reader(f)
    raw_documents = list(reader)

print("Prepare documents")
documents = [{'base_path': doc[0], 'text': doc[2]} for doc in raw_documents if doc[2] != '']

engine = GensimEngine(documents, log=True)
engine.train(number_of_topics=8)

print("Print topics to file")
topics_file = open('output/curated_early_years_topics.csv', 'w')
for topic in engine.topics:
    topic_string = "{},{}\n".format(topic['topic_id'], topic['words'])
    topics_file.write(topic_string)
topics_file.close()

print("Tagging documents")
untagged_documents = []
print("Reading input file 'input/early-years.csv'")
with open('input/early-years.csv', 'r') as f:
    reader = csv.reader(f)
    untagged_documents = list(reader)

print("Prepare documents")
untagged_documents = [{'base_path': doc[0], 'text': doc[1]} for doc in untagged_documents]

tagged_documents = engine.tag(untagged_documents)

print('Print tagged documents to file')
tagged_documents_file = open('output/curated_early_years_tagged_data.csv', 'w')
for tagged_document in tagged_documents:
    tag_string = '{},{}\n'.format(tagged_document['base_path'], tagged_document['tags'])
    tagged_documents_file.write(tag_string)
tagged_documents_file.close()
