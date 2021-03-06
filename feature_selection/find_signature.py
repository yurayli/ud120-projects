#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r") )
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, \
	test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### We want to use a overfit case to find signatures of each.
### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
accu = accuracy_score(pred, labels_test) # test accuracy
print "Trained data points:", len(labels_train)
print "Train accuracy:", clf.score(features_train, labels_train)
print "Test data points:", len(pred)
print "Test accuracy:", accu

featureImp = clf.feature_importances_
featureImp = numpy.array(featureImp)
pwrImp = featureImp[featureImp > 0.2]
maxImp = numpy.where(featureImp == max(featureImp))[0][0]
print "Maximum importance and its order in features:", max(featureImp), maxImp # 0.7647 33614; 0.6667 14343
print "Several power Imps:", pwrImp[:10] if len(pwrImp) >= 10 else pwrImp[0]
print "Most important word:", vectorizer.get_feature_names()[maxImp] # sshacklensf; cgermannsf
