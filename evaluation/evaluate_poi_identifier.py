#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here (overfitting case)
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
#pred = clf.predict(features_test)
acc = clf.score(features_test, labels_test) # test accuracy


print "Accuracy in train set:", clf.score(features_train, labels_train)
print "Accuracy in test set:", acc
#print "Num of people in test set:", len(labels_test)
#print "Num of POI in test set:", labels_test.count(1)
#print "Num of predicted POI in test set:", len(np.where(pred == 1)[0])

labels_test = np.array(labels_test)
p_index = np.where(labels_test == 1)[0] # for truly POI
n_index = np.where(labels_test == 0)[0] # for truly nonPOI
TP = len(np.where(labels_test[p_index] == pred[p_index])[0])
FN = len(np.where(labels_test[p_index] != pred[p_index])[0])
FP = len(np.where(labels_test[n_index] != pred[n_index])[0])
TN = len(np.where(labels_test[n_index] == pred[n_index])[0])
print "IN TEST SET"
print "Truly POI   :", [TP, FN]
print "Truly nonPOI:", [FP, TN]
P = TP / (TP+FP)
R = TP / (TP+FN)
print "Precision   :", P
print "Recall      :", R
try:
	F1 = 2*P*R / (P+R)
	print "F1 score    :", F1
except:
	pass

