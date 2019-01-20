import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import inspect 
import fns
from datetime import datetime

# number integers to round some outputs
n_round = 4

# MAIN
# Get data but only keep certified and denied outcomes 
# (future work could possibly include the merger of the other outcomes, i.e. certified expired as certified)
data = pd.read_csv("data/us_perm_visas.csv", nrows = 100,low_memory = False) #
data = data[data['case_status'].isin(['Certified', 'Denied'])]
pd.Series(data.columns).to_csv("all_cols.csv")
data['country_of_citizenship'].fillna(data['country_of_citzenship'], inplace = True)
data.drop('country_of_citzenship', inplace = True, axis = 1)
	
cols = data.columns

#~ for col in cols:
	#~ nan_frac = 1.*(len(data[col])- data[col].count())/(1.*len(data[col]))
	#~ if nan_frac == 1:
		#~ print col

data['decision_date_elapsed']  = pd.to_datetime(data['decision_date'],format = '%Y-%m-%d')
sorted_dates = data.sort_values(by = ['decision_date_elapsed'])['decision_date_elapsed']
start = sorted_dates.iloc[0]

data['decision_date_elapsed'] = data['decision_date_elapsed'].subtract( [start]*len(data) ).apply(lambda x: x.days)
