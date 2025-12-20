# ChatNoir Resiliparse

[![Build Wheels](https://github.com/chatnoir-eu/chatnoir-resiliparse/actions/workflows/build-wheels.yml/badge.svg)](https://github.com/chatnoir-eu/chatnoir-resiliparse/actions/workflows/build-wheels.yml)
[![Codecov](https://codecov.io/gh/chatnoir-eu/chatnoir-resiliparse/branch/develop/graph/badge.svg?token=VA51APYHU5)](https://codecov.io/gh/chatnoir-eu/chatnoir-resiliparse)
[![Documentation Status](https://readthedocs.org/projects/chatnoir-resiliparse/badge/?version=latest)](https://resiliparse.chatnoir.eu/en/latest/?badge=latest)

A collection of robust and fast processing tools for parsing and analyzing web archive data.

## Usage Instructions
For detailed information about the build process, dependencies, APIs, or usage instructions, please read the [Resiliparse Documentation](https://resiliparse.chatnoir.eu/en/latest/index.html)

## Resiliparse Module Summary
The Resiliparse main module with the following subcomponents:

### Parsing Utilities
The Resiliparse Parsing Utilities are highly optimized tools for dealing with encodings, detecting content types of raw protocol payloads, parsing HTML web pages, performing language detection, and more.

Main documentation: [Resiliparse Parsing Utilities](https://resiliparse.chatnoir.eu/en/latest/man/parse.html)

### Extraction Utilities
The Resiliparse Extraction Utilities are a set of performance-optimized and highly efficient tools for extracting structural or semantic information from noisy raw web data for further processing, such as (main) content extraction / boilerplate removal, schema extraction, general web data cleansing, and more.

Main documentation: [Resiliparse Extraction Utilities](https://resiliparse.chatnoir.eu/en/latest/man/extract.html)

### Process Guards
The Resiliparse Process Guard module is a set of decorators and context managers for guarding a processing context to stay within pre-defined limits for execution time and memory usage. Process Guards help to ensure the (partially) successful completion of batch processing jobs in which individual tasks may time out or use abnormal amounts of memory, but in which the success of the whole job is not threatened by (a few) individual failures. A guarded processing context will be interrupted upon exceeding its resource limits so that the task can be skipped or rescheduled.

Main documentation: [Resiliparse Process Guards](https://resiliparse.chatnoir.eu/en/latest/man/process-guard.html)

## Installation
The main Resiliparse package can be installed from PyPi as follows:
```bash
pip install resiliparse
```

## Building From Source
To build Resiliparse from sources, you need to install all required build-time dependencies first. On Ubuntu, this is done as follows:
```bash
# Add Lexbor repository
curl -sL https://lexbor.com/keys/lexbor_signing.key | \
  sudo gpg --dearmor --output /etc/apt/trusted.gpg.d/lexbor.gpg
echo "deb https://packages.lexbor.com/ubuntu/ $(lsb_release -sc) liblexbor" | \
    sudo tee /etc/apt/sources.list.d/lexbor.list

# Install build dependencies (requires libre2-dev>=2022-04-01)
sudo apt update
sudo apt install build-essential python3-dev zlib1g-dev \
    libuchardet-dev liblexbor-dev libre2-dev
```
Then, to build the package, run:
```bash
# Optional: Create a fresh venv first
python3 -m venv venv && source venv/bin/activate

# Build and install Resiliparse
pip install -e resiliparse_dom
```
Instead of building the package from this repository, you can also build it from the PyPi source package:
```bash
# Build Resiliparse from PyPi
pip install --no-binary resiliparse resiliparse
```

## Cite Us

Resiliparse is part of the [ChatNoir](https://chatnoir.eu/) web analytics toolkit. If you use ChatNoir or any of its tools for a publication, you can make us happy by citing our [ECIR 2018 demo paper](https://webis.de/downloads/publications/papers/bevendorff_2018.pdf):
```bibtex
@InProceedings{bevendorff:2018,
  address =             {Berlin Heidelberg New York},
  author =              {Janek Bevendorff and Benno Stein and Matthias Hagen and Martin Potthast},
  booktitle =           {Advances in Information Retrieval. 40th European Conference on IR Research (ECIR 2018)},
  editor =              {Leif Azzopardi and Allan Hanbury and Gabriella Pasi and Benjamin Piwowarski},
  month =               mar,
  publisher =           {Springer},
  series =              {Lecture Notes in Computer Science},
  site =                {Grenoble, France},
  title =               {{Elastic ChatNoir: Search Engine for the ClueWeb and the Common Crawl}},
  year =                2018
}
```
