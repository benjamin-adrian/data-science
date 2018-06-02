################################################
#####   EMPOLIS MACHINE LEARNING WORKSHOP  #####
###########     - MAR 22 2016 -    #############
################################################

This code showcases basic machine learning 
techniques as presented in the workshop.

Contact: Adrian Ulges (adrian.ulges@hs-rm.de)

### REQUIREMENTS ###############################

- The code has been tested with Python 2.7
- Required libraries (incl. tested version):
  - nltk (version 2.0b9)
  - pandas (0.17.1)
  - scipy (0.9.0)
  - sklearn (0.17)
  - numpy (1.110rc1)
  - matplotlib (1.1.1rc)

### CONTENTS ###################################

titanic/
	classifier_pandas.py  -> some basic code for
	                         basic feature engineering and
				 decision tree classification
				 on the Titanic Survival Prediction 
				 Problem
        train.csv
	test.csv	      -> training and test set
	tree.dot
	tree.pdf	      -> visualizations of the learned
			      	 decision tree

nytimes_data/
	train/*/*
	test/*/*	      -> news classification training and test
			      	 documents (crawled from nytimes.com).
				 Documents come in 8 categories
				 corresponding to 8 subfolders


classifiers/
	classifier.py	        -> main program for document classification
			      	   (using maximum entropy, decision trees,
				   or naive Bayes). Check with 'python classifier.py -h'
	preprocess_documents.py -> feature I/O and preprocessing (no need to touch)
	stopwords.txt	        -> stopwords list



clustering/kmeans/
	classifier.py	        -> main program for document clustering
			      	   (using K-Means). Check with 'python clusterer.py -h'
	preprocess_documents.py -> feature I/O and preprocessing (no need to touch)
	stopwords.txt	        -> stopwords list



active_learning/
	active_classifier.py    -> main program for active learning. 
				   Check with 'python active_classifier.py -h'
        classifier.py           -> the base classifier implementation (very similar to
				   'classifiers/classifier.py')
	preprocess_documents.py -> feature I/O and preprocessing (no need to touch)
	stopwords.txt	        -> stopwords list
	
	list_sports/
		labeled.txt     -> a mini training set to initialize active learning
		unlabeled.txt   -> lots of unlabeled samples to label manually during
				   the active learning process
		test.txt        -> test data to validate progress of active learning