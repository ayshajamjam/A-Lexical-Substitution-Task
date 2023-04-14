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

import re

import string

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
    # Get lemma, POS from context
    lemma = context.lemma
    pos = context.pos
    sentence = set(context.left_context + context.right_context)
    # print("Sentence: ", sentence)

    # Remove stop words in context sentence
    stop_words = set(stopwords.words('english'))
    sentence_non_stop_words = set()
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    for word in sentence:
        if word not in stop_words and word not in punc:
            sentence_non_stop_words.add(word.lower())
    
    # print("Sentence w/o Stop words: ", sentence_non_stop_words)

    # Iterate over all posible synsets target word appears in
    # Get candidates
    synsets = wn.synsets(lemma, pos)    # Get all synsets lemma appears in
    synsets_ovelap_count = defaultdict(int)
    word_count = defaultdict(int)

    for s in synsets:
        synsets_words = set()
        # Synset Definition
        # print(s)
        # print(s.definition())
        definition = tokenize(s.definition())
        # print('Definition: ', definition)
        synsets_words.update(definition)

        # Examples for synset
        for ex in s.examples():
            example = tokenize(ex)
            # print('Example: ', example)
            synsets_words.update(example)

        # Find all hypernyms
        hypernyms = set()
        # print(s.hypernyms())
        for h in s.hypernyms():
            # Definition of each hypernym
            # print(h.definition())
            hyper_def = tokenize(h.definition())
            # print('Hypernym definition: ', hyper_def)
            synsets_words.update(hyper_def)
            
            # Examples for each hypernym
            # print(h.examples())
            for ex in h.examples():
                example = tokenize(ex)
                # print('Hypernym example: ', example)
                synsets_words.update(example)

        synsets_non_stop_words = set()

        # Remove all stop words and punc
        for word in synsets_words:
            if word not in stop_words and word not in punc:
                synsets_non_stop_words.add(word)

        # print("All possible words: ", synsets_non_stop_words)

        # find overlap between synsets_non_stop_words and sentence_non_stop_words
        overlap = synsets_non_stop_words.intersection(sentence_non_stop_words)
        a = len(overlap)
        # print("Overlap: ", overlap)
        synsets_ovelap_count[s] += a

        b = 0  # The frequency (.count) of <q,s>
        for l in s.lemmas():
            # print(l)
            word = str(l.name()).replace('_', ' ').lower()
            # print(word, ' ---> ', l.count())
            if(str(word) == str(lemma)):
                b = l.count()
            else:
                c = l.count()
                word_count[word] += 1000 * a + c
        for w in word_count.keys():
            word_count[w] += 100 * b

    # print(word_count)
    if(len(word_count) > 0):
        # print(max(word_count, key=word_count.get), ' ==> ', max(word_count.values()))
        return max(word_count, key=word_count.get)
    else:
        print('NONE')
        return None

    # max_synset = max(synsets_ovelap_count, key=synsets_ovelap_count.get)
    # max_synset_count = max(synsets_ovelap_count.values())
    # print(max_synset, ' ---> ', max_synset_count)
    # print(synsets_ovelap_count)
    # # print(max_synset.lemmas())

    # # Determine synset with the highest score
    # weighted_score = 0

    # # No overlap between context and all synsets
    # if(max_synset_count == 0):
    #     # print("OVERLAP")
    #     selected_word = wn_frequency_predictor(context)
    #     # print(selected_word)
    #     return selected_word
    # # Some overlap but there is a tie
    # list_synsets_with_same_count = [k for k,v in synsets_ovelap_count.items() if int(v) == max_synset_count]
    # if(len(list_synsets_with_same_count) > 0):
    #     # print("TIEEEE")
    #     # print(list_synsets_with_same_count)
        
    #     wordCounts = defaultdict(int)       # dict keeps track of word count accross synsets
    #     # Get lemmas that appear in these synsets
    #     for s in list_synsets_with_same_count:
    #         for l in s.lemmas():
    #             # l = Lemma('boring.s.01.dull') 
    #             word = str(l.name()).replace('_', ' ').lower()
    #             # print(word, l.count())
    #             if(str(word) != str(lemma)):
    #                 wordCounts[word] += l.count()

    #     # print(wordCounts)
    #     if(len(wordCounts) == 0):
    #         selected_word = wn_frequency_predictor(context)
    #         # print(selected_word)
    #         return selected_word
    #     else:
    #         selected_word = max(wordCounts, key=wordCounts.get)
    #         # print(selected_word)
    #         return selected_word
        
    # # Clear winner
    # wordCounts = defaultdict(int)
    # for l in max_synset.lemmas():
    #     # l = Lemma('boring.s.01.dull') 
    #     word = str(l.name()).replace('_', ' ').lower()
    #     # print(word, l.count())
    #     if(str(word) != str(lemma)):
    #         wordCounts[word] += l.count()
    
    # if(len(wordCounts) > 0):
    #     selected_word = max(wordCounts, key=wordCounts.get)
    #     return selected_word
    # else:
    #     return #TODO - return second best synset
    # # print(selected_word)
    # return selected_word #replace for part 3        
   
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
        # print('\n')
        # # print("Context")
        # print(context)  # useful for debugging
        prediction = wn_simple_lesk_predictor(context)
        # prediction = smurf_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
