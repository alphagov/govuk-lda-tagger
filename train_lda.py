"""
Train an LDA model on a CSV file containing "url" and "text" columns.
"""
from __future__ import print_function
import argparse
from gensim_engine import GensimEngine
from evaluation import ModelEvaluator
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
    '--output-dictionary', dest='output_dict', metavar='FILENAME',
    help="Filename to save the dictionary to. This can be loaded in next time using --input-dictionary, to speed up the process."
)
parser.add_argument(
    '--nobigrams', dest='bigrams', action='store_false',
    help="Don't include bigrams in the model's vocabulary."
)
parser.add_argument(
    '--numtopics', dest='number_of_topics', type=int, default=None,
    help="Number of topics to train"
)
parser.add_argument(
    '--mintopics', dest='minimum_topics', type=int, default=1,
    help="Minimum number of topics to train"
)
parser.add_argument(
    '--maxtopics', dest='maximum_topics', type=int, default=25,
    help="Maximum number of topics to train"
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

    if args.output_dict:
        engine.save_dictionary(args.output_dict)

    training_options = dict(
        words_per_topic=args.words_per_topic,
        passes=args.passes
    )

    if args.number_of_topics is None:
        engine.train_best_number_of_topics(
            ModelEvaluator(engine.corpus),
            min_topics=args.minimum_topics,
            max_topics=args.maximum_topics + 1,
            **training_options
        )
    else:
        engine.train(number_of_topics=args.number_of_topics, **training_options)

    if args.topics_filename:
        print("Exporting topics to {}".format(args.topics_filename))
        export_topics(engine.topics, args.topics_filename)

    if args.tags_filename:
        print("Exporting tags to {}".format(args.tags_filename))
        tags = engine.tag(training_documents)
        export_tags(tags, args.tags_filename)

    if args.vis_filename:
        print("Exporting visualisation to {}".format(args.vis_filename))
        engine.visualise(args.vis_filename)
