## Try different algorithms on author_id data

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

t0 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print "preprocessing time:", round(time() - t0, 3), "s"


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

t1 = time()
#clf = KNeighborsClassifier(n_neighbors=7)
#clf = AdaBoostClassifier()
clf = RandomForestClassifier()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t1, 3), "s"

t2 = time()
pred = clf.predict(features_test) # testing
print "testing time:", round(time() - t2, 3), "s"

accu = accuracy_score(pred, labels_test) # test accuracy, 93.8% for knn w/ k=5, 95.11% for AdaBoost and 99.2 for RandomForest
print "Accuracy:", accu
print "Whole elapsed time:", round(time() - t1, 3), "s"