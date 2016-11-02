#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
	"""
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    #cleaned_data = []
	### your code goes here
	resid_err = abs(predictions - net_worths)
	
	resid_err = resid_err.T[0]
	ages = ages.T[0]
	net_worths = net_worths.T[0]
	for i in range( int(round(0.1 * len(ages))) ):
		which = np.where(resid_err == max(resid_err))[0][0]
		resid_err = np.delete(resid_err, which)
		ages = np.delete(ages, which)
		net_worths = np.delete(net_worths, which)
    
	cleaned_data = zip(ages, net_worths, resid_err)
    
	return cleaned_data

