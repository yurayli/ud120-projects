#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"] # salary, long_term_incentive, and more..
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)


### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, target_train)

print "\nSlope:", reg.coef_
print "Intercept:", reg.intercept_
print "\n #### status on train dataset ####"
print "r-squared score:", reg.score(feature_train, target_train)
print "\n #### status on test dataset ####"
print "r-squared score:", reg.score(feature_test, target_test)


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
plt.scatter( feature_test, target_test, color="r", label="test" ) 
plt.scatter( feature_train, target_train, color="b", label="train" ) 
#plt.plot( feature_train, reg.predict(feature_train), color="c")
plt.plot( feature_test, reg.predict(feature_test), color="g", label="predictOfTest (fit train)")

# interchange training set and test set
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="c", label="predictOfTrain (fit test)")
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.savefig("enron.png")
plt.show()


##
print "\n -------- interchange train set and test set --------- \n"
print "New Slope:", reg.coef_
print "New Intercept:", reg.intercept_
print "\n #### status on new train dataset ####"
print "r-squared score:", reg.score(feature_train, target_train)
print "\n #### status on new test dataset ####"
print "r-squared score:", reg.score(feature_test, target_test)
print "--------------------------------- \n"