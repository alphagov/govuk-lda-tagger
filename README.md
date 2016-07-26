# Tag documents with the LDA algorithm

An experiment of using the LDA machine learning algorithm to generate topics
from documents and tag them with those topics

## Install requirements

These scripts were run on python `3.5.2`. You can install it via `pyenv` and
`python-build`.

In order to install the requirements, run `pip install -r requirements.txt`.

## Run

The scripts have to be run in sequence:

1. `python generate_titles.py` - this will generate a file in `output` called
   `data.titles`, which includes all the titles of the documents.
2. `python generate_tokens.py` - this will generate a file in `output` called
   `data.tokens`, which includes all the relevant tokens from the documents.
3. `python generate_ldac.py` - this will generate a file in `output` called
   `data.ldac`, which includes all the token count information.
4. `python generate_topics.py` - this will generate a file in `output` called
   `data.txt`, which includes all documents tagged and also a number of topics
   used to tagged those documents.
