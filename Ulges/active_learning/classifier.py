#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

from preprocess_documents import read_and_preprocess_documents
import sys
import argparse
import pickle
import os
from math import log
from operator import itemgetter
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

"""

  Text Classification
  -------------------------
  a small interface for English document classification.

  Train the classifier on the training documents like this:

  > python classifier.py --train nytimes_data/train/*/*

  Apply the classifier to the test documents like this:

  > python classifier.py --apply nytimes_data/test/*/*

"""

class Classifier:

    def __init__(self):

        """ The classifier should store all its learned information
            in this 'model' object. Pick whatever form seems appropriate
            to you. """
        self.model = None

    def train(self, features, labels):
        """
        trains a document classifier and stores all relevant
        information in 'self.model'.

        @type features: dict
        @param features: Each entry in 'features' represents a document
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
        @type labels: dict
        @param labels: 'labels' contains the class labels for all documents
                       in dictionary form:
                       {
                           'doc1.txt': 'arts',
                           'doc2.txt': 'business',
                           'doc3.txt': 'sports',
                           ...
                       }
        """
        raise NotImplementedError()


    def apply(self, features):
        """
        applies a classifier to a set of documents. Requires the classifier
        to be trained (i.e., you need to call train() before you can call test()).

        @type features: dict
        @param features: Each entry in 'features' represents a document
                         by its (sparse) bag-of-words vector. 'features'
                         is of the following form (i.e., for each document, 
                         all terms occurring in the document and their
                         counts are stored):
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
        @rtype: dict
        @return: For each document classified, apply() returns the label
                 of the class the document has been assigned to. The return value
                 is a dictionary of the form:
                 {
                   'doc1.txt': 'arts',
                   'doc2.txt': 'travel',
                   'doc3.txt': 'sports',
                   ...
                 }
        """
        raise NotImplementedError()




class ClassifierSKLearn(Classifier):

    def __init__(self, model,
                 alpha=1.0, binarize=0.5, max_depth=20):
        Classifier.__init__(self)
        self.model_file = 'model_' + model + '_sklearn.pkl'
        self.model = model

        # choose classifier model
        if model=='naive_bayes':
            self.classifier = BernoulliNB(alpha=alpha, binarize=binarize)
        elif model=='dectree':
            self.classifier = DecisionTreeClassifier(max_depth=22)
        elif model=='maxent':
            self.classifier = LogisticRegression()
        else:
            raise ValueError("Invalid classifier: " + model + ".")

        # if possible: load model
        if os.path.isfile(self.model_file):
            with open(self.model_file, "r") as stream:
                (self.classifier,self.vectorizer,self.int2class,self.class2int) = pickle.load(stream)


    def train(self, features, labels):

        # transform samples into sklearn format
        self.vectorizer = DictVectorizer(sparse=True)
        X = self.vectorizer.fit_transform(features.values())

        # transform labels to sklearn format
        classes = sorted(set(labels.values()))
        self.class2int = dict(zip(classes,range(len(classes))))
        self.int2class = dict(zip(range(len(classes)),classes))
        y = [self.class2int[labels[filename]] for filename in features]

        # train model
        self.classifier = self.classifier.fit(X, y)

        # store model
        with open(self.model_file, "w") as stream:
            pickle.dump((self.classifier,self.vectorizer,self.int2class,self.class2int), stream)


    def apply(self, features):

        X = self.vectorizer.transform(features.values())
        labels = self.classifier.predict(X)

        result = {}
        for filename,label in zip(features.keys(),labels):
            result[filename] = self.int2class[label]

        return result

    def score(self, features, category):

        X = self.vectorizer.transform(features.values())
        scores = self.classifier.predict_proba(X)

        result = {}
        for filename,sc in zip(features.keys(),scores):
            result[filename] = sc[self.class2int[category]]

        return result


    def inspect(self):
        if self.model=='naive_bayes':
            raise NotImplementedError()
        elif self.model=='dectree':
            self._inspect_dectree()
        elif self.model=='maxent':
            self._inspect_maxent()
        
    def _inspect_dectree(self):
        with open("tree.dot", 'w') as f:
            f = tree.export_graphviz(self.classifier, out_file=f)
            print ("Wrote decision tree to 'tree.dot'. Print to PDF using \n"\
                       "  > dot -Tpdf tree.dot -o tree.pdf")
        
    def _inspect_maxent(self):
        terms = self.vectorizer.get_feature_names()
        for i,c in self.int2class.items():
            weights = self.classifier.coef_[i,:]
            best_features = sorted(enumerate(weights), key=itemgetter(1), reverse=True)[:10]
            best_terms = [terms[j] for j,val in best_features]
            print 'class', c
            for t in best_terms:
                print '   ', t


def error_rate(result, labels):
    errors = 0
    for filename,l in labels.items():
        if filename not in result:
            print 'WARNING! File ' + filename + ' not in result!'
        errors += (result.get(filename,-1) != l)
    return float(errors)/len(labels)

if __name__ == "__main__":

    # parse command line arguments (no need to touch)
    parser = argparse.ArgumentParser(description='A text classifier based on Naive Bayes.')
    parser.add_argument('mode', type=str,
                        help='which classifier to apply ("maxent", "naive_bayes", "dectree"')
    parser.add_argument('documents', metavar='doc', type=str, nargs='+',
                        help='documents to train/apply the classifier on/to')
    parser.add_argument('--train', help="train the classifier", action='store_true')
    parser.add_argument('--apply', help="apply the classifier (you'll need to train or load"\
                                        "a trained model first)", action='store_true')
    parser.add_argument('--inspect', help="get some info about the learned model", action='store_true')
    args = parser.parse_args()

    # reads and preprocesses the documents listed as commandline arguments. 
    # You can use the resulting features for classification.
    wordmap, features = read_and_preprocess_documents(args.documents)

    # estimate class labels ('arts', 'business', 'dining', ...)
    # from directory names
    labels = {}
    for filename in features:
        tokens = filename.split("/")
        classlabel = tokens[-2]
        labels[filename] = classlabel

    classifier = ClassifierSKLearn(args.mode)

    #  train classifier on 'features' and 'labels' 
    # (using documents from the 'train' folder)
    if args.train:
        classifier.train(features, labels)

    # apply the classifier to documents from
    # the 'test' folder
    if args.apply:
        result = classifier.apply(features)
        for filename,label in result.items():
            print filename, label
        print 'Error rate: %.3f' %error_rate(result, labels)

    # get some info about the learned model
    if args.inspect:
        classifier.inspect()
