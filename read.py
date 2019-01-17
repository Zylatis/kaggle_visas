import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz 
from difflib import get_close_matches
# number integers to round some outputs
n_round = 4

# Get data but only keep certified and denied outcomes 
# (future work could possibly include the merger of the other outcomes, i.e. certified expired as certified)
data = pd.read_csv("data/us_perm_visas.csv", nrows = 15000, low_memory = False) #
data = data[data['case_status'].isin(['Certified', 'Denied'])]

hr_to_year = 2080

#~ data =  data[['case_status','us_economic_sector','class_of_admission','wage_offer_from_9089']].dropna()
drop_cols = ['pw_soc_code', 'case_no','pw_level_9089','naics_2007_us_code','wage_offer_from_9089', 'employer_address_1', "pw_job_title_9089", "wage_offer_unit_of_pay_9089"]
data.drop(drop_cols, axis = 1, inplace = True)
cols = data.columns

# Aggresive removal of missing data:
# if a col is nan for >90% of the data then drop it so dropping
# rows with *any* nans doesn't wipe out all the data. It is insufficient
# to do just with len(data with dropped) !=0, still kills everything 
non_na_cols = []
for col in cols:
	if len(data[col].dropna()) >  0.9*len(data[col]):
		non_na_cols.append(col)

# Keep only the columns where >90% of the data are not nan, then drop the rows
# which have *ANY* entry as nan
prev = len(data)
data = data[non_na_cols].dropna()

# see how many and which survived
print "Old v new number of cols kept"
print len(cols), len(data.columns)
print "Old v new number of rows kept"
print prev, len(data)

# convert prevailing wage to salary (some are p/a some are p/hr)
data['annual_salary'] = ""
for index, row in data.iterrows():
	if row['pw_unit_of_pay_9089'] == 'yr':
		data.set_value(index, 'annual_salary', row['pw_amount_9089'])
	else:
		data.set_value(index, 'annual_salary', hr_to_year*row['pw_amount_9089'])

# drop the target variable from training set
inp = data.drop('case_status',axis = 1) 

# define target variable
out = data['case_status']

# Do one-hot transformation on the categorical data. 
# This is super cool and basically a way to make sure we have binary values for the categories
# rather than label by number, which will bugger up the data/scaling
one_hot_data = pd.get_dummies(inp,drop_first=True)
train_x, test_x, train_y, test_y = train_test_split(one_hot_data, out)
ntrain = len(train_x)

# Define and run tree classifier
clf = tree.DecisionTreeClassifier(max_leaf_nodes = 100) 
clf = clf.fit(train_x, train_y)
predict = clf.predict(train_x)
train_accuracy = accuracy_score(train_y, predict)

# check against validation set
predict = clf.predict(test_x)
test_accuracy = accuracy_score(test_y, predict)

print "Train/test accuracy:"
print round(train_accuracy,n_round), round(test_accuracy, n_round) 
print "Number of nodes in tree"
print clf.tree_.node_count
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=one_hot_data.columns, class_names=out, filled=True, rounded=True,  special_characters=True, leaves_parallel = True) 
graph = graphviz.Source(dot_data) 
graph.render("visa_tree")
