#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-



################################################
#####   EMPOLIS MACHINE LEARNING WORKSHOP  #####
###########     - MAR 22 2016 -    #############
################################################

"""
This code showcases basic machine learning 
techniques as presented in the workshop.

Contact: Adrian Ulges (adrian.ulges@hs-rm.de)
"""


import operator
import nltk
import sys
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

"""
  !!!!!!!!!!!!!!!!!!!!
  !!! DO NOT TOUCH !!!
  !!!!!!!!!!!!!!!!!!!!

  a bit of code to read and preprocess articles
  from the NYTimes dataset. Includes:
  - tokenization
  - stemming
  - filtering terms by their frequency

  See 'read_and_preprocess_documents()' for the actual interface.
  All other code is for internal use only.
"""


def read_stopwords(filename):
    with open(filename) as stream:
        lines = stream.readlines()
    stopwords = [l.rstrip() for l in lines]
    return stopwords
        

STOPWORDS = set(read_stopwords("stopwords.txt"))


def read_article(filename):
    with open(filename) as stream:
        lines = stream.readlines()
        url,date,title,intro,story = lines[0],lines[1],lines[2],lines[3],' '.join(lines[5:])
        return title,intro,story

def read_article_tokens(filename):
    title,intro,story = read_article(filename)
    txt = ' '.join([title,intro,story])
    tokens = preprocess(txt)
    return tokens

def preprocess(txt):
    """
    tokenizes the input text, lowercases it, removes stopwords, 
    and does a stemming using the well-known Porter stemmer.

    @type txt: str
    @param txt: the input text to be preprocessed
    @rtype: list(str)
    @return: a list of tokens
    """
    txt = txt.replace('.', ' . ')
    txt = txt.lower()

    tokens = nltk.word_tokenize(txt)

    tokens = [t for t in tokens if len(t)>1]
    tokens = [t for t in tokens if t not in STOPWORDS]

    porter = nltk.PorterStemmer()
    #for t in tokens:
    #    print t, type(t)
    tokens = [porter.stem(t) for t in tokens]

    return tokens



def build_wordmap(all_files, min_frequency=10, min_doc_frequency=5):
    """
    given a document collection ('all_files'), this method
    builds a dictionary containing term counts.

    @type all_files: list(str)
    @param all_files: a list of document filenames
    @type min_frequency: int
    @param min_frequency: terms with a total count less than this threshold
                          are ommitted from the result.
    @type min_doc_frequency: int
    @param min_doc_frequency: terms occurring in fewer documents than this threshold
                          are ommitted from the result.
    @rtype: dict
    @return: a dictionary wordmap mapping all terms (remaining after preprocessing and filtering)
             to a unique ID, of the form:
             {
               term1: 1,
               term2: 2,
               ...
             }
    """
    all_words = {}

    for i,filename in enumerate(all_files):

       words_before = len(all_words)

       tokens = read_article_tokens(filename)

       # count all term occurrences in this document in a dictionary
       count = {}
       [count.__setitem__(t,1+count.get(t,0)) for t in tokens]

       # add this document's term occurrences to global term count
       for t in count:
         n,m = all_words.get(t, (0,0))
         all_words[t] = (n+1, m+count[t])

       print ( "After %d (of %d) documents: Have %d distinct terms (%d new terms)!" 
               %(i+1, len(all_files), len(all_words), len(all_words)-words_before) )

    # filter words by frequency
    for (w,(n,m)) in all_words.items():
       if m < min_frequency:
           del all_words[w]
       elif n < min_doc_frequency:
           del all_words[w]

    print ( "After filtering by frequency (occurrences >= %d, occurrences_in_docs >= %d): Have %d terms left."
           %(min_frequency, min_doc_frequency, len(all_words)) )

    sorted_words = sorted(all_words.iteritems(), key=operator.itemgetter(1), reverse=True)
    wordmap = dict([(w,i) for i,(w,freq) in enumerate(sorted_words)])
      
    return wordmap


def get_bag_of_words(filename, wordmap, cleartext):
  
    tokens = read_article_tokens(filename)
    bag_of_words = {}

    for t in tokens:
        if t in wordmap:
            idx = t if cleartext else wordmap[t] 
            bag_of_words[idx] = bag_of_words.get(idx, 0) + 1

    return bag_of_words


def read_and_preprocess_documents(all_files, cleartext=True, min_frequency=10, min_doc_frequency=5):
    """
    given a list of filenames, read and preprocess all text from the files,
    filter the resulting termset, and return bag-of-words representations
    for all files in the list.

    @type all_files: list(str)
    @param all_files: a list of document filenames
    @type cleartext: boolean
    @param cleartext: whether to use actual terms (i.e., strings) as keys in the result
                      or term IDs.
    @type min_frequency: int
    @param min_frequency: terms with a total count less than this threshold
                          are ommitted from the result.
    @type min_doc_frequency: int
    @param min_doc_frequency: terms occurring in fewer documents than this threshold
                          are ommitted from the result.

    @rtype: tuple
    @return: the return value is of the form (wordmap,features), with ...
             - wordmap: a dictionary of all terms, mapping each term to an ID, of the form
                        {
                          'the': 1,
                          'and': 2,
                          ...
                        }
             - features: The bag-of-word features extracted for each input file.
                         A dictionary of the form
                         {
                           'doc1.txt':
                              {
                                'the' : 17,
                                'world': 3, 
                                ...
                              },
                           'doc2.txt':
                              {
                                'community' : 2,
                                'college': 1, 
                                ...
                              },
                            ...
                         }
    """
    wordmap = build_wordmap(all_files, min_frequency, min_doc_frequency)

    features = {}

    for filename in all_files:
        bag_of_words = get_bag_of_words(filename, wordmap, cleartext)
        features[filename] = bag_of_words

    return wordmap, features


