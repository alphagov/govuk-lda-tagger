# Tag documents with the LDA algorithm

This repo has a bunch of scrpits to try tag a number of documents based on the
LDA algorithm.

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
2. `python generate_ldac.py` - this will generate a file in `output` called
   `data.ldac`, which includes all the token count information.

TODO
