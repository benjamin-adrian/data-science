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


import sys
import argparse
from preprocess_documents import read_and_preprocess_documents
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans

"""
  Text Clustering
  -------------------------
  A K-Means clusterer (basically, a wrapper for sklearn's
  KMeans implementation).

  Run clustering on the training documents like this:

  > python clusterer.py nytimes_data/train/*/*
"""



class KMeansClusterer:

    def __init__(self, K, runs=10):
        """
        constructor for the K-Means object

        @type K: int
        @param K: the number of clusters to estimate (must be > 0)
        @type runs: int
        @param runs: the number of K-Means restarts (in the end
                     the best result is picked)
        """
        self.K = K
        self.runs = runs
        self.kmeans = None                  # sklearn KMeans object
        self.vectorizer = DictVectorizer()  # sklearn vectorizer (transforms input data
                                            # into sklearn's internal format)

    def train(self, features):
        """
        clusters a set of text documents, represented by bag-of-words features

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
        """
        self.kmeans = KMeans(n_clusters = self.K,
                             init = 'random',
                             n_init = self.runs,
                             verbose = 1)
        features = self.vectorizer.fit_transform(features.values())
        # features /= features.sum(axis=1)
        self.kmeans.fit(features)


    def inspect(self, features, narticles=15):
        """
            prints the top-N articles for each cluster
            (the 'top' articles are the ones closest
            to the cluster center)
        """
        _features = self.vectorizer.transform(features.values())

        # compute distance for each article to each cluster center
        dists = self.kmeans.transform(_features).T

        # for each cluster, print the articles closest to the cluster center
        for k,d in enumerate(dists):
            best = d.argsort()[:narticles]
            print '\n', 'Cluster', k
            for i in best:
                print '  ', features.keys()[i]



if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='A text clusterer based on K-Means.')
    parser.add_argument('documents', metavar='doc', type=str, nargs='+',
                        help='documents to apply the clusterer to')
    args = parser.parse_args()

    # reads and preprocesses the documents listed as commandline arguments. 
    # You can use the resulting features for classification.
    wordmap, features = read_and_preprocess_documents(args.documents)

    # fixed number of clusters: 30
    kmeans = KMeansClusterer(30)

    #  run clustering
    kmeans.train(features)

    # print sample articles for each cluster
    kmeans.inspect(features)
