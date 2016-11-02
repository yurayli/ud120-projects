#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from addFeature import addFeature

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','fraction_to_poi','fraction_from_poi','shared_receipt_with_poi',
				 'bonus','total_stock_value']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    my_dataset = pickle.load(data_file)

### Task 2: Remove outliers
my_dataset.pop('TOTAL')
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = addFeature(my_dataset)

### Extract features and labels from dataset for local testing
### Scale the features
from sklearn.preprocessing import MinMaxScaler
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)
print "Num of features:", len(features_list)-1
print "Num of data points:", len(labels)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

clf_list = [GaussianNB(), SVC(), KNeighborsClassifier(n_neighbors=5, weights='distance'),
			DecisionTreeClassifier(min_samples_split=5, max_depth=4, max_leaf_nodes=15),
			AdaBoostClassifier(), RandomForestClassifier()]
clf = clf_list[2]


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

crs_validate = True

if crs_validate:
	
	t0 = time()
	cv = KFold( len(labels), 5, shuffle=True )
	accu_train_list = []
	accu_list = []
	P_list = []
	R_list = []
	F1_list = []
	for train_indices, test_indices in cv:
	
		features_train = [features[ii] for ii in train_indices]
		features_test  = [features[ii] for ii in test_indices]
		labels_train   = [labels[ii] for ii in train_indices]
		labels_test    = [labels[ii] for ii in test_indices]
		
		clf.fit(features_train, labels_train)
		pred = clf.predict(features_test)
		accu_train_list.append( clf.score(features_train,labels_train) )
		accu_list.append( clf.score(features_test,labels_test) )
		
		labels_test = np.array(labels_test)
		p_index = np.where(labels_test == 1)[0] # for truly POI
		n_index = np.where(labels_test == 0)[0] # for truly nonPOI
		TP = len(np.where(labels_test[p_index] == pred[p_index])[0])
		FN = len(np.where(labels_test[p_index] != pred[p_index])[0])
		FP = len(np.where(labels_test[n_index] != pred[n_index])[0])
		TN = len(np.where(labels_test[n_index] == pred[n_index])[0])
		try:
			P = TP / float(TP+FP)
			P_list.append(P)
		except ZeroDivisionError:
			pass
		try:
			R = TP / float(TP+FN)
			R_list.append(R)
		except ZeroDivisionError:
			pass
		try:
			F1 = 2*P*R / (P+R)
			F1_list.append(F1)
		except:
			pass

	print "training and validating time:", round(time() - t0, 3), "s"
	print "Avg train accuracy:", np.mean(accu_train_list)
	print "Avg accuracy      :", np.mean(accu_list)
	print "Avg precision     :", np.mean(P_list)
	print "Avg recall        :", np.mean(R_list)
	print "Avg F1            :", np.mean(F1_list)


else:
	
	features_train, features_test, labels_train, labels_test = \
		train_test_split(features, labels, test_size=0.2, random_state=42)
	t0 = time()
	clf.fit(features_train, labels_train)
	pred = clf.predict(features_test) # testing
	print "training and validating time:", round(time() - t0, 3), "s"

	accu = accuracy_score(pred, labels_test) # test accuracy
	print "Train accuracy:", clf.score(features_train, labels_train)
	print "Test accuracy:", accu

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
	try:
		P = TP / float(TP+FP)
		print "Precision   :", P
	except ZeroDivisionError:
		pass
	try:
		R = TP / float(TP+FN)
		print "Recall      :", R
	except ZeroDivisionError:
		pass
	try:
		F1 = 2*P*R / (P+R)
		print "F1 score    :", F1
	except:
		pass
	


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)