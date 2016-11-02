from numpy import *

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=5)
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