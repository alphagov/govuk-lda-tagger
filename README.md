# Tag documents with the LDA algorithm

An experiment of using the LDA machine learning algorithm to generate topics
from documents and tag them with those topics

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

### Example scripts

If you are only interested in looking at some quick results, there are 2 scripts
you could run.

#### Using Python's lda library

Run `python run_lda.py` in order to use the LDA library to generate topics and
categorise the documents listed in the input file.

#### Using Python's gensim library

Run `python run_gensim.py` in order to use the gensim library to generate topics
and categorise the documents listed in the input file.

### Using the GensimEngine

For more advanced usage, there is a class that can help. In `gensim_engine.py`
there is a class that can be used to train and run an LDA model.

There are a few scripts that make use of it:
- `early_years.py` - Using the early years data from the HTML pages to derive
  topics;
- `early_years_curated.py` - Using the early years' audit content as training
  data and the search API content of 700 documents to tag;
- `early_years_titles_descriptions.py` - Using the early years' audit content as
  training data and the search API titles and descriptions of 700 documents to
  tag.

All these scripts make use of the following API:

```
# Instantiate an object
engine = GensimEngine(documents, log=True)

# Train the model with the data provided
engine.train(number_of_topics=20)

# Tag a bunch of untagged documments
engine.tag(untagged_documents)
```

Each document is expected to be a list of dictionaries, where each dictionary
has a `base_path` key and a `text` key.

## Fetching new data

### Import indexable content from the search API

### Import PDF data

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
