## Try different algorithms on author_id data

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

t0 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print "preprocessing time:", round(time() - t0, 3), "s"
# features_train.shape  (15820, 379)
# features_test.shape  (1758, 379)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


#clf = KNeighborsClassifier(n_neighbors=7)
#clf = AdaBoostClassifier()
#clf = GradientBoostingClassifier()
clf = RandomForestClassifier(n_estimators=100)
clf.fit(features_train, labels_train)
print "Train accuracy:", clf.score(features_train, labels_train)

print "no. of testing emails:", len(labels_test)
pred = clf.predict(features_test) # testing 
accu = accuracy_score(pred, labels_test) # test accuracy, 93.8% for knn w/ k=5, 95.11% for AdaBoost and 99.2 for RandomForest
print "Test accuracy:", accu

