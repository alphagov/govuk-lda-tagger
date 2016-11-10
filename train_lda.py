"""
Train an LDA model on a CSV file containing "url" and "text" columns.
"""
from __future__ import print_function
import argparse
import datetime
import os
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
import_parser.add_argument(
    '--no-below', dest='no_below', type=int, default=0,
    help="Filter out words that occur less than this number of times in the corpus."
)
import_parser.add_argument(
    '--no-above', dest='no_above', type=float, default=0.5,
    help="Filter out words that make up more than this fraction of the corpus."
)
import_parser.add_argument(
    '--keep-n', dest='keep_n', type=int, default=None,
    help="Keep this many terms in the dictionary after filtering extremes."
)
import_parser.add_argument(
    '--experiment', dest='experiment', default=None,
    help="Name of experiment"
)
refine_parser.add_argument(
    'experiment', metavar='EXPERIMENT',
    help='Name of a previous experiment, eg 2016-11-01_15-44-06_695357'
)


parser.add_argument(
    '--output-topics', dest='topics_filename', metavar='FILENAME', default=None,
    help='Save topics data to a file.'
)
parser.add_argument(
    '--output-tags', dest='tags_filename', metavar='FILENAME', default=None,
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
    '--vis-filename', dest='vis_filename', metavar='FILENAME', default=None,
    help="Save visualisation of the topics to a file."
)
parser.add_argument(
    '--use-phrasemachine', dest='use_phrasemachine', action='store_true',
    help="Use phrasemachine instead of lemmatization when building the dictionary."
)
parser.add_argument(
    '--use-textacy', dest='use_textacy', action='store_true',
    help='Use textacy to generate bigrams/noun phrases'
)
parser.add_argument(
    '--no-lemmatisation', dest='use_lemmatisation', action='store_false',
    help='Use lemmatisation to refine input text'
)
parser.add_argument(
    '--use-tfidf', dest='use_tfidf', action='store_true',
    help="Weight terms in a document according to TF-IDF."
)

if __name__ == '__main__':
    args = parser.parse_args()

    experiment_name = args.experiment
    if experiment_name is None:
        experiment_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')

    experiment_path = os.path.join('experiments', experiment_name)

    if args.command == 'import':
        os.makedirs(experiment_path)
        os.makedirs(os.path.join(experiment_path, 'models'))

        print("Loading input file {}".format(args.training_documents))
        training_documents = load_documents(args.training_documents)

        engine = GensimEngine.from_documents(
            training_documents,
            log=True,
            dictionary_path=args.dictionary,
            include_bigrams=args.bigrams,
            use_phrasemachine=args.use_phrasemachine,
            use_textacy=args.use_textacy,
            use_lemmatisation=args.use_lemmatisation,
            use_tfidf=args.use_tfidf,
            no_below=args.no_below,
            no_above=args.no_above,
            keep_n=args.keep_n,
        )

    else:
        print("Loading experiment {}".format(experiment_name))
        engine = GensimEngine.from_experiment(experiment_name, log=True)

    print("Training...")
    experiment = engine.train(number_of_topics=args.number_of_topics, words_per_topic=args.words_per_topic, passes=args.passes)

    print('Saving experiment: {}'.format(experiment_name))
    experiment.save(experiment_name)

    topics_filename = args.topics_filename or os.path.join(experiment_path, 'topics')
    print("Exporting topics to {}".format(topics_filename))
    export_topics(engine.topics, topics_filename)

    tags_filename = args.tags_filename or os.path.join(experiment_path, 'tags')
    print("Exporting tags to {}".format(tags_filename))
    tags = experiment.tag()
    export_tags(tags, tags_filename)

    vis_filename = args.vis_filename or os.path.join(experiment_path, 'vis.html')
    print("Exporting visualisation to {}".format(vis_filename))
    experiment.visualise(vis_filename)
