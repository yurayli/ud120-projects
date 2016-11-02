## This file is to try some pieces of the main (poi_id.py)
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import math
import numpy as np


# visualize data distribution with features
enron = pickle.load( open("final_project_dataset.pkl", "r") )
enron.pop('TOTAL')
feature_list = ['poi','shared_receipt_with_poi','fraction_to_poi']
data = featureFormat(enron, feature_list)

p_index = np.where(data[0:,0] == 1.)[0]
n_index = np.where(data[0:,0] == 0.)[0]
plt.scatter(data[p_index,1], data[p_index,2], color='r')
plt.scatter(data[n_index,1], data[n_index,2], color='b')

plt.xlabel("shared_receipt_with_poi")
plt.ylabel("fraction_to_poi")
plt.show()


# count missing values
count = 0
for name in names:
	if math.isnan(float(enron[name]['total_stock_value'])):
		count += 1
