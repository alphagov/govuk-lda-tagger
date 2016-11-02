"""
Train an LDA model on a CSV file containing "url" and "text" columns.
"""
from __future__ import print_function
import argparse
import datetime
from gensim_engine import GensimEngine
from model_io import load_documents, export_topics, export_tags


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    'training_documents', metavar='FILENAME',
    help='File containing the training documents'
)
parser.add_argument(
    '--input-dictionary', dest='dictionary', metavar='FILENAME',
    help='A curated dictionary file. If not specified, the dictionary will be generated from the training documents.'
)
parser.add_argument(
    '--output-topics', dest='topics_filename', metavar='FILENAME',
    help='Save topics data to a file.'
)
parser.add_argument(
    '--output-tags', dest='tags_filename', metavar='FILENAME',
    help='Save tagged documents to a file.'
)
parser.add_argument(
    '--nobigrams', dest='bigrams', action='store_false',
    help="Don't include bigrams in the model's vocabulary."
)
parser.add_argument(
    '--numtopics', dest='number_of_topics', type=int, default=20,
    help="Number of topics to train"
)
parser.add_argument(
    '--words-per-topic', dest='words_per_topic', type=int, default=8,
    help="Words per topic"
)
parser.add_argument(
    '--passes', dest='passes', type=int, default=50,
    help="Number of LDA passes"
)
parser.add_argument(
    '--vis-filename', dest='vis_filename', metavar='FILENAME',
    help="Save visualisation of the topics to a file."
)

if __name__ == '__main__':
    args = parser.parse_args()

    print("Loading input file {}".format(args.training_documents))
    training_documents = load_documents(args.training_documents)

    print("Training...")
    engine = GensimEngine(
        training_documents,
        log=True,
        dictionary_path=args.dictionary,
        include_bigrams=args.bigrams,
    )

    experiment = engine.train(number_of_topics=args.number_of_topics, words_per_topic=args.words_per_topic, passes=args.passes)

    name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    print('Saving experiment: {}'.format(name))
    experiment.save(name)

    if args.topics_filename:
        print("Exporting topics to {}".format(args.topics_filename))
        export_topics(engine.topics, args.topics_filename)

    if args.tags_filename:
        print("Exporting tags to {}".format(args.tags_filename))
        tags = engine.tag(training_documents)
        export_tags(tags, args.tags_filename)

    if args.vis_filename:
        print("Exporting visualisation to {}".format(args.vis_filename))
        experiment.visualise(args.vis_filename)
