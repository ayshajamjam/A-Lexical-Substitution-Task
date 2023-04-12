#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

from collections import defaultdict

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1

    # Get all synsets lemma appears in
    synsets = wn.synsets(lemma, pos)
    words = set()
    # Get lemmas that appear in these synsets
    for s in synsets:
        for l in s.lemmas():
            # l = Lemma('boring.s.01.dull') 
            word = str(l.name()).replace('_', ' ').lower()
            if(str(word) != str(lemma)):
                words.add(word)
    return words

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:

    # Get lemma, POS from context
    lemma = context.lemma
    pos = context.pos
    # print(lemma)
    # print(pos)

    # Get candidates
    synsets = wn.synsets(lemma, pos)    # Get all synsets lemma appears in
    wordCounts = defaultdict(int)       # dict keeps track of word count accross synsets
    # Get lemmas that appear in these synsets
    for s in synsets:
        for l in s.lemmas():
            # l = Lemma('boring.s.01.dull') 
            word = str(l.name()).replace('_', ' ').lower()
            # print(word, l.count())
            if(str(word) != str(lemma)):
                wordCounts[word] += l.count()

    # print(wordCounts)
    # print(max(wordCounts, key=wordCounts.get))

    return max(wordCounts, key=wordCounts.get) # select highest count as substitute

def wn_simple_lesk_predictor(context : Context) -> str:
    return None #replace for part 3        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        return None # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        return None # replace for part 5

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    # get_candidates('spin', 'v')

    for context in read_lexsub_xml(sys.argv[1]):
        # print(context)  # useful for debugging
        prediction = wn_frequency_predictor(context)
        # prediction = smurf_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
