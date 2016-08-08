# Tag documents with the LDA algorithm

An experiment of using the LDA machine learning algorithm to generate topics
from documents and tag them with those topics

## Install requirements

These scripts were run on python `2.7.12`. You can install it via `pyenv` and
`python-build`. One of the dependencies is incompatible with python 3.

In order to install the requirements, run `pip install -r requirements.txt`.

### Note on dependencies

Please note that one of the dependencies requires a `Fortran` compiler installed
in your machine.

I used gfortran from http://hpc.sourceforge.net/. in order to install it, I
downloaded the specific version of gfortran for my OS version and installed it:

```
gunzip gfortran-5.1-bin.tar.gz
sudo tar -xvf gcc gfortran-5.1-bin.tar -C /
```

This will copy the binary and other files into their expected lcoation.

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
