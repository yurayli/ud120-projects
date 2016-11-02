#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import numpy as np
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
t0 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print "preprocessing time:", round(time() - t0, 3), "s"


#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]


## First finding the best C for optimized SVM
Cparams = [10.0, 1e2, 1e3, 1e4]
bestC = 10.0
bestAccu = 0
t1 = time()
for C in Cparams:
	clf = SVC(kernel="rbf", C = C)
	cv = KFold( len(labels_train), 5, shuffle=True )
	accu_valid_list = []
	for train_indices, valid_indices in cv:
		features		= [features_train[ii] for ii in train_indices]
		features_valid  = [features_train[ii] for ii in valid_indices]
		labels		    = [labels_train[ii] for ii in train_indices]
		labels_valid    = [labels_train[ii] for ii in valid_indices]
		
		clf.fit(features, labels)
		pred = clf.predict(features_valid)
		accu_valid_list.append( accuracy_score(pred, labels_valid) )

	accu_valid = np.mean(accu_valid_list)
	if accu_valid > bestAccu:
		bestAccu = accu_valid
		bestC = C

print "optimizing time:", round(time() - t1, 3), "s"	

t2 = time()
clf = SVC(kernel="rbf", C = bestC)
clf.fit(features_train, labels_train) # training
print "training time:", round(time() - t2, 3), "s"

t3 = time()
pred = clf.predict(features_test) # testing
print "testing time:", round(time() - t3, 3), "s"

accu = accuracy_score(pred, labels_test) # test accuracy 97.61%
print "Accuracy:", accu

print "Whole elapsed time:", round(time() - t0, 3), "s"
print "The (10, 26, 50)th labels:", pred[10], pred[26], pred[50]
print "How many are predicted to be in the 'Chris' (1) class?", list(pred).count(1)
#########################################################


