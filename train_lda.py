"""
Train an LDA model on a CSV file containing "url" and "text" columns.
"""
from __future__ import print_function
import argparse
import datetime
from gensim_engine import GensimEngine
from model_io import load_documents, export_topics, export_tags

parser = argparse.ArgumentParser(description=__doc__)
subparsers = parser.add_subparsers(help='sub-commands')

import_parser = subparsers.add_parser('import', help='Import a new dataset')
import_parser.set_defaults(command='import')

refine_parser = subparsers.add_parser('refine', help='Retrain an existing experiment')
refine_parser.set_defaults(command='refine')

import_parser.add_argument(
    'training_documents', metavar='CSV',
    help='File containing the training documents'
)
import_parser.add_argument(
    '--input-dictionary', dest='dictionary', metavar='DICTIONARY',
    help='A curated dictionary file. If not specified, the dictionary will be generated from the training documents.'
)
import_parser.add_argument(
    '--nobigrams', dest='bigrams', action='store_false',
    help="Don't include bigrams in the model's vocabulary."
)

refine_parser.add_argument(
    'experiment', metavar='EXPERIMENT',
    help='Name of a previous experiment, eg 2016-11-01_15-44-06_695357'
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
parser.add_argument(
    '--use-phrasemachine', dest='use_phrasemachine', action='store_true',
    help="Use phrasemachine instead of lemmatization when building the dictionary."
)
parser.add_argument(
    '--use-tfidf', dest='use_tfidf', action='store_true',
    help="Weight terms in a document according to TF-IDF."
)
if __name__ == '__main__':
    args = parser.parse_args()

    if args.command == 'import':
        print("Loading input file {}".format(args.training_documents))
        training_documents = load_documents(args.training_documents)

        engine = GensimEngine.from_documents(
            training_documents,
            log=True,
            dictionary_path=args.dictionary,
            include_bigrams=args.bigrams,
            use_phrasemachine=args.use_phrasemachine,
            use_tfidf=args.use_tfidf
        )

    else:
        print ("Loading experiment {}".format(args.experiment))
        engine = GensimEngine.from_experiment(args.experiment, log=True)

    print("Training...")
    experiment = engine.train(number_of_topics=args.number_of_topics, words_per_topic=args.words_per_topic, passes=args.passes)

    name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    print('Saving experiment: {}'.format(name))
    experiment.save(name)

    if args.topics_filename:
        print("Exporting topics to {}".format(args.topics_filename))
        export_topics(engine.topics, args.topics_filename)

    if args.tags_filename:
        print("Exporting tags to {}".format(args.tags_filename))
        tags = experiment.tag()
        export_tags(tags, args.tags_filename)

    if args.vis_filename:
        print("Exporting visualisation to {}".format(args.vis_filename))
        experiment.visualise(args.vis_filename)
