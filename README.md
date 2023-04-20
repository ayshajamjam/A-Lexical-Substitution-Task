# A Lexical Substitution Task

## Introduction

In this assignment you will work on a lexical substitution task, using, WordNet, pre-trained Word2Vec embeddings, and BERT. This task was first proposed as a shared task at SemEval 2007 Task 10. In this task, the goal is to find lexical substitutes for individual target words in context. For example, given the following sentence:

"Anyway , my pants are getting tighter every day ." 

the goal is to propose an alternative word for tight, such that the meaning of the sentence is preserved. Such a substitute could be constricting, small or uncomfortable.

In the sentence

"If your money is tight don't cut corners ." 

the substitute small would not fit, and instead possible substitutes include scarce, sparse, limitited, constricted. You will implement a number of basic approaches to this problem and compare their performance.

## Setup

1. python3 -m venv env
2. source env/bin/activate
3. pip install nltk
4. pip install --upgrade gensim
5. pip install transformers
6. pip install tensorflow


## Libraries

1. NLTK- The standard way to access WordNet in Python is now NLTK, the Natural Language Toolkit. NLTK contains a number of useful resources, such as POS taggers and parsers, as well as access to several text corpora and other data sets. In this assignment you will mostly use its WordNet interface. 
2. Gensim- Gensim is a vector space modeling package for Python. While gensim includes a complete implementation of word2vec (among other approaches), we will use it only to load existing word embeddings. To install gensim, try
3. Huggingface Transformers- We will use the BERT implementation by Huggingface (an NLP company), or more specifically their slightly more compact model DistilBERT.

## Pre-trained Word2Vec Embeddings

These embeddings were trained using a modified skip-gram architecture on 100B words of Google News text, with a context window of +-5. The word embeddings have 300 dimensions. We will not train a BERT model ourselves, but we will experiment with the pre-trained masked language model to find substitutes. 

## Files

- lexsub_trial.xml - input trial data containing 300 sentences with a single target word each.
- gold.trial - gold annotations for the trial data (substitues for each word suggested by 5 judges).
- lexsub_xml.py - an XML parser that reads lexsub_trial.xml into Python objects.
- lexsub_main.py - this is the main scaffolding code you will complete
- score.pl - the scoring script provided for the SemEval 2007 lexical substitution task.


### Part 2

- python lexsub_main.py lexsub_trial.xml  > smurf2.predict
- perl score.pl smurf2.predict gold.trial

### Part 3

- python lexsub_main.py lexsub_trial.xml  > smurf3.predict
- perl score.pl smurf3.predict gold.trial

### Part 4

- python lexsub_main.py lexsub_trial.xml  > smurf4.predict
- perl score.pl smurf4.predict gold.trial

### Part 5

- python lexsub_main.py lexsub_trial.xml  > smurf5.predict
- perl score.pl smurf5.predict gold.trial

### Part 6
- python lexsub_main.py lexsub_trial.xml  > smurf6.predict
- perl score.pl smurf6.predict gold.trial