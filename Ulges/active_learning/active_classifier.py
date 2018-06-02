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


from preprocess_documents import read_and_preprocess_documents
import sys
import argparse
import pickle
import random
import os
from math import log
from operator import itemgetter
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from classifier import ClassifierSKLearn

"""
  Text Classification with Active Learning
  ----------------------------------------
  A binary text classifier (e.g., 'sports' vs. 'nosports')
  that learns interactively.Performs a training on a set of 
  documents that are initially unlabeled (except a few 
  initial samples).
  Iteratively, the classifier selects samples for you to label,
  collects your feedback, and improves classification.
  Improvements of classification are tracked by validating on
  a separate test set.

  Sample call:

  > python classifier.py uncertainty maxent sports lists_sports/labeled.txt \
                         lists_sports/unlabeled.txt lists_sports/test.txt

"""


class ActiveClassifier:

    def __init__(self, category, strategy, classifier):

        """ The classifier should store all its learned information
            in this 'model' object. Pick whatever form seems appropriate
            to you. """
        self.classifier = classifier
        self.category = category
        self.strategy = strategy


    def binarize_labels(self, labels):
        """ turns category labels ('sports', 'business', 'arts', ...)
            into binary labels ('sports', 'no_sports') """
        return dict([(f,l if l==self.category else 'no_'+self.category) for (f,l) in labels.items()])
    

    def retrain(self, features, labels):
        self.classifier.train(features, labels)

        
    def validate(self, features, labels):
        """ checks classifier's error rate on held-out
            test set and prints result """
        def error_rate(result, labels):
            errors = 0
            for filename,l in labels.items():
                if filename not in result:
                    print 'WARNING! File ' + filename + ' not in result!'
                errors += (result.get(filename,-1) != l)
            return float(errors)/len(labels)

        result = self.classifier.apply(features)
        print 'Error rate: %.3f' %error_rate(result, labels)


    def select(self, features):
        """ main active learning step: select a sample to label! """
        scores_ = self.classifier.score(features, self.category)
        scores = [(f,scores_[f]) for f in features]
        # select sampling strategy
        if self.strategy == 'uncertainty':
            return self._select_uncertainty(scores)
        elif self.strategy == 'random':
            return self._select_random(scores)
        elif self.strategy == 'relevance':
            return self._select_relevance(scores)
        else:
            raise ValueError('Unknown sample selection method! ' + self.strategy)

    def _select_uncertainty(self, scores):
        scores = sorted(scores, key=lambda (f,s): abs(0.5-s))
        return scores[0][0]

    def _select_random(self, scores):
        scores = sorted(scores, key=lambda (f,s): random.random())
        return scores[0][0]

    def _select_relevance(self, scores):
        scores = sorted(scores, key=lambda (f,s): s, reverse=True)
        return scores[0][0]


    def label_manually(self, filename):
        """ collect a manual label from the shell """
        print '>>>', filename
        while True:
            l = raw_input(">>> Does this belong to '%s'? (1/0) " %self.category)
            try: 
                l = int(l)
                if l==1:
                    return self.category
                elif l==0:
                    return 'no_'+self.category
                else:
                    print;print; "Bad input! Must be '1' or '0'!"; print
            except:
                    print;print; "Bad input! Must be '1' or '0'!"; print
                

    def active_train(self, 
                     features_labeled, labels_labeled, 
                     features_unlabeled, 
                     features_test, labels_test):
        """
        Main active learning method. Performs an interative training
        (in which more and more 'interesting' samples are labeled
        and the model is improved).
        
        Each entry in any 'features*' input collection represents a document
        by its (sparse) bag-of-words vector. 'features'
        is of the following form (i.e., for each document, 
        all terms occurring in the document and their
        counts are stored in a dictionary):
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
         'labels*' contains the class labels for documents
         in dictionary form:
                       {
                           'doc1.txt': 'arts',
                           'doc2.txt': 'business',
                           'doc3.txt': 'sports',
                           ...
                       }
        @type features_labeled: dict
        @param features_labeled: Features of a few initial labeled training samples
                              (see format above)
        @type labels_labeled: dict
        @param labels_labeled: labels of initial labeled training samples
                              (see format above)
        @type features_unlabeled: dict
        @param features_unlabeled: Initially, many features are unlabeled.
                                   We will label these in the course of active learning.
        @type features_test: dict
        @type features_test: A held-out test set to validate progress of active learning.
        @type labels_test: dict
        @param labels_init: labels of held-out test set
                            (see format above)

        """
        labels_labeled = self.binarize_labels(labels_labeled)
        labels_test    = self.binarize_labels(labels_test)

        while True:
            # retrain the base classifier
            self.retrain(features_labeled, labels_labeled)

            # check error on held-out test set
            self.validate(features_test, labels_test)

            # pick a sample and label it
            sample_filename = self.select(features_unlabeled)
            label = self.label_manually(sample_filename)
            
            # add labeled data to labeled set and remove from unlabeled seet
            features_labeled[sample_filename] = features_unlabeled[sample_filename]
            labels_labeled[sample_filename] = label
            del features_unlabeled[sample_filename]




def parse_list(filename):
    """ parse a list of input files and returns a Python list of filenames """
    with open(filename) as stream:
        lines = stream.readlines()
        return [l.rstrip('\n') for l in lines]


if __name__ == "__main__":

    # parse command line arguments (no need to touch)
    parser = argparse.ArgumentParser(description='A text classifier based on Naive Bayes.')
    parser.add_argument('strategy', type=str,
                        help='active learning strategy ("uncertainty", "random", "relevance")')
    parser.add_argument('model', type=str,
                        help='which classifier to apply ("maxent", "naive_bayes", "dectree")')
    parser.add_argument('category', type=str,
                        help='which category to learn ("business", "travel", "sports", ...)')
    parser.add_argument('labeled', type=str,
                        help='list of a few labeled files (to bootstrap learning)')
    parser.add_argument('unlabeled', type=str,
                        help='list of unlabeled files (will be labeled during active learning)')
    parser.add_argument('test', type=str,
                        help='list of test files (to validate progress of learning)')
    args = parser.parse_args()

    # read input documents
    labeled_files   = parse_list(args.labeled)
    unlabeled_files = parse_list(args.unlabeled)
    test_files      = parse_list(args.test)

    # reads and preprocesses the documents listed as commandline arguments. 
    # You can use the resulting features for classification.
    all_files = labeled_files + unlabeled_files + test_files
    wordmap, features = read_and_preprocess_documents(all_files)

    # estimate class labels ('arts', 'business', 'dining', ...)
    # from directory names
    labels = {}
    labels_test = {}
    for filename in labeled_files:
        tokens = filename.split("/")
        classlabel = tokens[-2]
        labels[filename] = classlabel
    for filename in test_files:
        tokens = filename.split("/")
        classlabel = tokens[-2]
        labels_test[filename] = classlabel

    base_classifier = ClassifierSKLearn(args.model)
    classifier = ActiveClassifier(args.category, args.strategy, base_classifier)

    features_unlabeled = dict([(f,features[f]) for f in unlabeled_files])
    features_labeled   = dict([(f,features[f]) for f in labeled_files])
    features_test      = dict([(f,features[f]) for f in test_files])

    classifier.active_train(features_labeled, labels, 
                            features_unlabeled, 
                            features_test, labels_test)

