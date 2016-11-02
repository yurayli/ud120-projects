from numpy import *
from sklearn.naive_bayes import GaussianNB


def classify(features_train, labels_train):
    ### create classifier with GaussianNB
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf


def submitAccuracy(clf, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    
    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)

    ### calculate and return the accuracy on the test data
    #ytest = array(labels_test)
    accuracy = mean((pred == labels_test).astype(float))
    
    # Or
    # from sklearn.metrics import accuracy_score
    # print accuracy_score(pred, labels_test)
    
    return accuracy
    


