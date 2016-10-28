"""
Train an LDA model on a CSV file containing "url" and "text" columns.
"""
from __future__ import print_function
import argparse
from gensim_engine import GensimEngine
from model_io import load_documents, export_topics, export_tags


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_filename', metavar='FILENAME',
                    help='Input file')
parser.add_argument('--topics-filename', dest='topics_filename', metavar='FILENAME',
                    help='Topics output file')
parser.add_argument('--tags-filename', dest='tags_filename', metavar='FILENAME',
                    help='Tags output file')
parser.add_argument('--curated-dictionary', dest='dictionary', metavar='FILENAME',
                    help='Optional curated dictionary file')
parser.add_argument('--nobigrams', dest='bigrams', action='store_false', help="Don't include bigrams in the model's vocabulary.")
parser.add_argument('--output-dictionary-filename', dest='output_dict')


if __name__ == '__main__':
    args = parser.parse_args()

    print("Loading input file {}".format(args.input_filename))
    documents = load_documents(args.input_filename)

    print("Training...")
    engine = GensimEngine(
        documents,
        log=True,
        dictionary_path=args.dictionary,
        include_bigrams=args.bigrams,
    )
    engine.train(dictionary_save_path=args.output_dict)

    if args.topics_filename:
        print("Exporting topics to {}".format(args.topics_filename))
        export_topics(engine.topics, args.topics_filename)

    if args.tags_filename:
        print("Exporting tags to {}".format(args.tags_filename))
        tags = engine.tag(documents)
        export_tags(tags, args.tags_filename)
