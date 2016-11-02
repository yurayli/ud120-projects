import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )
features = ["bonus", "long_term_incentive"]
data = featureFormat(data_dict, features)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)


### do PCA and show the dimension reduced data
def doPCA(data):
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)
	pca.fit(data)
	return pca

pca = doPCA(data)
print pca.explained_variance_ratio_
first_pc = pca.components_[0]
second_pc = pca.components_[1]

trans_data = pca.transform(data)
mu_shift = [np.mean(data[0:,0] - trans_data[0:,0]), np.mean(data[0:,1] - trans_data[0:,1])]
for ii, jj in zip(trans_data, data):
	plt.scatter( first_pc[0]*ii[0] + mu_shift[0], first_pc[1]*ii[0] + mu_shift[1], color="r" )
	plt.scatter( second_pc[0]*ii[1] + mu_shift[0], second_pc[1]*ii[1] + mu_shift[1], color="c" )
	plt.scatter( jj[0], jj[1], color="b" )

plt.xlabel("rescaled bonus")
plt.ylabel("rescaled long_term incentive")
plt.axis('image')
#plt.xlim(-5e6, 1e7)
#plt.ylim(-5e6, 1e7)
plt.savefig('enron_pca.png')
plt.show()
