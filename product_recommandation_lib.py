import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.misc import derivative

def scale_features(X, scaler_type='Standard'):
	"""
	X is a DataFrame
	Possible scaler_type values: 'Standard', 'Minmax'
	returns X as a scaled DataFrame with its scaler
	"""
	if scaler_type == 'Standard':
		scaler = StandardScaler()
	elif scaler_type == 'MinMax':
		scaler = MinMaxScaler(feature_range=(0, 1))

	scaler.fit(X)
	columns = X.columns
	X = scaler.transform(X)
	X = pd.DataFrame(data=X,columns=columns)
	return X, scaler

def apk(actual, predicted, k=10):
	"""
	Computes the average precision at k.
	This function computes the average prescision at k between two lists of
	items.
	Parameters
	----------
	actual : list
			 A list of elements that are to be predicted (order doesn't matter)
	predicted : list
				A list of predicted elements (order does matter)
	k : int, optional
		The maximum number of predicted elements
	Returns
	-------
	score : double
			The average precision at k over the input lists
	"""
	if len(predicted)>k:
		predicted = predicted[:k]

	score = 0.0
	num_hits = 0.0

	for i,p in enumerate(predicted):
		if p in actual and p not in predicted[:i]:
			num_hits += 1.0
			score += num_hits / (i+1.0)

	if not actual:
		return 0.0

	return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
	"""
	Computes the mean average precision at k.
	This function computes the mean average prescision at k between two lists
	of lists of items.
	Parameters
	----------
	actual : list
			 A list of lists of elements that are to be predicted
			 (order doesn't matter in the lists)
	predicted : list
				A list of lists of predicted elements
				(order matters in the lists)
	k : int, optional
		The maximum number of predicted elements
	Returns
	-------
	score : double
			The mean average precision at k over the input lists
	"""
	return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def get_features_importance(model):
	gain = model.feature_importance('gain')
	feat_imp = pd.DataFrame(
		{
			'feature':model.feature_name(), 
			'split':model.feature_importance('split'), 
			'gain':100 * gain / gain.sum()
		}
	).sort_values('gain', ascending=False)
	return feat_imp