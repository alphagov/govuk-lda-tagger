# Tag GOV.UK documents with the LDA algorithm

This project contains several experiments that used the [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) machine learning algorithm to generate topics from pages on [GOV.UK](https://www.gov.uk) and tag them with those topics.

## Results

The output of each experiment is in the `experiments` directory.

[You can open any of the ipython notebooks in nbviewer](https://nbviewer.jupyter.org/github/alphagov/govuk-lda-tagger/tree/master/experiments/).

Example results: [Education theme - all audits - all data excluding PDF](https://nbviewer.jupyter.org/github/alphagov/govuk-lda-tagger/blob/master/experiments/20_topics_without_pdf_data_tfidf/tfidf.ipynb#topic=0&lambda=0.21&term=)

## Nomenclature

- **Document**: a chunk of text representing a page on GOV.UK.
- **Base path**: The relative URL to a page on GOV.UK.
- **Corpus**: a set of documents.
- **Term**: a single word, phrase, or [n-gram](https://en.wikipedia.org/wiki/N-gram). We break a document into many terms before running the LDA algorithm.
- **Dictionary**: a data structure that maps every term to an integer ID.
- **Stopwords**: terms we want the algorithm to ignore - these won't be included in the dictionary.
- **Document term matrix**: [a data structure that captures how frequently terms appear in different documents](https://en.wikipedia.org/wiki/Document-term_matrix).
- **TF-IDF**: [Term Frequency - Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). A measure that shows how important a word is to a document in a corpus.
- **LDA**: [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) - the algorithm we're using to model topics.

## Install requirements

### Python version

These scripts were run on python `2.7.12`. You can install it via `pyenv` and
`python-build`. One of the dependencies is incompatible with python 3.

### Pre-requisites

Before you can install the requirements, please make sure you have a fortran
compiler on your system. More on that below..

I used gfortran from http://hpc.sourceforge.net/. in order to install it, I
downloaded the specific version of gfortran for my OS version and installed it:

```
gunzip gfortran-5.1-bin.tar.gz
sudo tar -xvf gcc gfortran-5.1-bin.tar -C /
```

This will copy the binary and other files into their expected lcoation.

### Installing python dependencies

Once that's done, run `pip install -r requirements.txt` in order to install all
the python dependencies.

### Post-install

We use a python library called `nltk` for natural language processing. We need a
module from `nltk` that doesn't come bundled with the library. In order to
install that module do the following:

1) Open a python console

```
$ python
Python 2.7.12 (default, Jun 29 2016, 14:05:02)
[GCC 4.2.1 Compatible Apple LLVM 7.3.0 (clang-703.0.31)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
```

2) Import `nltk` and open its package application:

```
>>> import nltk
>>> nltk.download()
showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
```

3) On the GUI it opened, click on `corpora` and scroll down until you find a
package named `stopwords`. Download that package and exit the app.

At this stage you will have a working setup to run the scripts below.

## Try it out

The `train_lda.py` script is a command line interface (CLI) to the LDA tagger. You can customise the input dataset, the preprocessing, and the parameters passed to the underlying LDA library.

### Generating topics and tags for early years

Using the early years data from the HTML pages to derive topics, and tagging every document to those topics:

```
train_lda.py import --experiment early_years input/early-years.csv
```

The `--experiment` option defines the output directory under `experiments`. It defaults to one generated from the current time.

### Using a curated dictionary

Pass a curated dictionary using the `--input-dictionary` option. By default the dictionary is generated from the corpus, excluding a number of predefined stopwords (defined in the `stopwords` directory).

```
train_lda.py import input/audits_with_content.csv --input-dictionary input/dictionary.txt
```

### Retraining using the same corpus

If you already ran an experiment, but something went wrong, you can use the `refine` subcommand to train it again, but reuse the corpus generated in the first run. The final argument is the original experiment directory name, which will be overwritten.

```
train_lda.py --numtopics 100 refine early-years
```

### Using the GensimEngine class

In `gensim_engine.py` there is a class that can be used to train and run an LDA model programatically.

This has the following API:

```
# Instantiate an object
engine = GensimEngine(documents, log=True)

# Train the model with the data provided
experiment = engine.train(number_of_topics=20)

# Tag all documents in the corpus
tags = experiment.tag()
```

`documents` is expected to be a list of dictionaries, where each dictionary has a `base_path` key and a `text` key.

### Other scripts
When we started the project we created two simple scripts to test the libraries we used.

You can run either of these to see some sample topics.

#### Using Python's lda library

Run `python run_lda.py` in order to use the LDA library to generate topics and categorise the documents listed in the input file.

#### Using Python's gensim library

Run `python run_gensim.py` in order to use the gensim library to generate topics and categorise the documents listed in the input file.


## Existing Data

| Filename        | Type           | Source  |
| ------------- |-------------| -----|
| input/all_audits_for_education.csv      | URLs with source audit | 2016 education audits |
| input/audits_with_content.csv      | URL, Text, Audit  | 2016 education audits |
| input/bigrams.csv | Bigram dictionary | Curated  |
| input/dictionary.txt | Term Dictionary | Lemmatisation/bigrams for audits_with_content.csv |
| input/early-years-audit-all-content.csv | Raw data | 2016 eudcation audit spreadsheet |
| input/early-years-titles-descriptions.csv | URL, Text | Titles and descriptions of early years audit content |
| input/early-years.csv | URL, Text | Content store text of early years audit content |
| input/running-a-school-audit.csv | URL, Text | Content store text of running a school audit content |
| expanded_audits/all_audits_for_education.csv | url,link,title,description,content,topics,organisations | Search API data for 2016 education audits |
| expanded_audits/all_audits_for_education_words_nopdf.csv | URL, Text | Same as above, with all text combined. |
| expanded_audits/all_audits_for_education_with_pdf_data.csv | URL, PDF data | Scraped PDF files from 2016 education audit |
| expanded_audits/all_audits_for_education_with_pdf_and_indexable_content.csv | URL, text | Combination of above two files |

## Fetching new data

### Import indexable content from the search API

In order to fetch data from the search API, prepare a CSV input file containing
one column (with the `URL` header) and the `base_path` of the links we wish to
fetch content for.

Then run the following command:

```
python import_indexable_content.py --environemnt https://www.gov.uk input_file.csv
```

This script outputs CSV rows with the title, description, indexable content,
topic names and organisation names.

### Import PDF data

In order to fetch PDF text from a number of GOV.UK base paths, prepare a CSV
input file containing one column (with the `URL` header) and the `base_path` of
the links we wish to fetch content for.

Then, run the following command:

```
python fetch_pdf_content.py input_file.csv output_file.csv
```

The output file will include the same base paths and also the text found in all
PDF attachments, merged into one big string.

### Combine all the data

The python tool [CSVKit](https://csvkit.readthedocs.io/en/0.9.1/index.html) can be used to combine the separate CSVs into one:

Note that because the columns are very wide, you will need to increase the default maximum field size:

```
csvjoin -c url all_audits_for_education.csv all_audits_for_education_with_pdf_data.csv > all_audits_for_education_with_pdf_and_indexable_content.csv --maxfieldsize [a big number]
```

The resulting CSV can then be passed to `data_import/combine_csv_columns.py` to merge everything into one "words" column.

```
python data_import/combine_csv_columns.py < all_audits_for_education_with_pdf_and_indexable_content.csv > all_audits_for_education_words.csv
```

## Licence

[MIT License](LICENCE.txt)
