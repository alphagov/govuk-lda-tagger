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

## Run

These repo has 2 scripts using different approaches.

### Using Python's lda library

Run `python run_lda.py` in order to use the LDA library to generate topics and
categorise the documents listed in the input file.

### Using Python's gensim library

Run `python run_gensim.py` in order to use the gensim library to generate topics
and categorise the documents listed in the input file.
