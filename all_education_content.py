import sys
import csv
import json
from gensim_engine import GensimEngine

csv.field_size_limit(sys.maxsize)

raw_documents = []
filename = "expanded_audits/all_audits_for_education_words.csv"
args = sys.argv[1:] if __name__ == '__main__' else ()

print("Reading input file")
with open(filename, 'r') as f:
    reader = csv.reader(f)
    raw_documents = list(reader)

print("Prepare documents")
documents = [{'base_path': doc[0], 'text': doc[1]} for doc in raw_documents if len(doc) == 2 and doc[1] != '']

engine = GensimEngine(documents, log=True, args=args)
engine.train(number_of_topics=10, passes=10)

print("Print topics to file")
topics_file = open('output/all_audits_for_education_words.csv', 'w')
for topic in engine.topics:
    topic_string = "{},{}\n".format(topic['topic_id'], topic['words'])
    topics_file.write(topic_string)
topics_file.close()

engine.visualise()
