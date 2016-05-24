#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
t0 = time()
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print "preprocessing time:", round(time() - t0, 3), "s"


#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

t1 = time()
clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
print "training time:", round(time() - t1, 3), "s"

t2 = time()
pred = clf.predict(features_test) # testing
print "testing time:", round(time() - t2, 3), "s"

accu = accuracy_score(pred, labels_test) # test accuracy
print "Accuracy:", accu

print "Whole elapsed time:", round(time() - t0, 3), "s"
print "Number of features:", len(features_train[0])
#########################################################


