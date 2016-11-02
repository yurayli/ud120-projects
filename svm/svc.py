import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()
print "Shape of examples", np.shape(np.array(features_train))

########################## SVM #################################
### we handle the import statement and SVC creation for you here
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

""" Optimizing params
Cparams = [.1, 10.0, 1e2, 1e3, 1e4]
Gamm = [.01, .1, 1]
for gamma in Gamm:
	for C in Cparams:
		t1 = time()
		clf = SVC(kernel="rbf", C=C, gamma=gamma)
		clf.fit(features_train, labels_train)
		print "training time:", round(time() - t1, 3), "s"

		#### store your predictions in a list named pred
		t2 = time()
		pred = clf.predict(features_test)
		print "testing time:", round(time() - t2, 3), "s"
		acc = accuracy_score(pred, labels_test)
		print "Accuracy:", acc
		print "Whole elapsed time:", round(time() - t1, 3), "s"
"""

t1 = time()
clf = SVC(kernel="rbf", C=1e4, gamma=1)
clf.fit(features_train, labels_train)
print "training time:", round(time() - t1, 3), "s"

#### store your predictions in a list named pred
t2 = time()
pred = clf.predict(features_test)
print "testing time:", round(time() - t2, 3), "s"
acc = accuracy_score(pred, labels_test) # 94.8%
print "Accuracy:", acc
print "Whole elapsed time:", round(time() - t1, 3), "s"


prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())