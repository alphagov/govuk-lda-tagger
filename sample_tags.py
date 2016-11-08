"""
This script reads the tag output and generates some markdown for sample tags,
for us to put in the jupyter notebook for the experiment.
"""
import sys
import csv
import fileinput
import ast
import random

reader = csv.reader(sys.stdin)
rows = list(fileinput.input())

sample = random.sample(rows, 20)

for row in sample:
    link, rest = row.strip().split(',', 1)
    print u'### {}'.format(link)
    tags = ast.literal_eval(rest)
    for topic_id, prob in tags:
        print u'- Topic {} ({:.0f}%)'.format(topic_id, prob * 100)
    print u''

