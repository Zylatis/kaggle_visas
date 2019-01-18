import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz 
from difflib import get_close_matches
import inspect 
# number integers to round some outputs
min_leaf_samples = 50 # interesting: if this is large then the tree only has 'certified' nodes which gives 85% accuracy
criterion = "entropy" # "gini"
def single_tree():
	# Define and run tree classifier
	clf = tree.DecisionTreeClassifier(min_samples_leaf = min_leaf_samples, criterion = criterion) #max_leaf_nodes 
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
	graph.render("single_tree")

def forest():
	# Define and run tree classifier
	clf = ensemble.RandomForestClassifier( n_estimators=10 ,min_samples_leaf = min_leaf_samples, criterion = criterion) #max_leaf_nodes 
	clf = clf.fit(train_x, train_y)
	predict = clf.predict(train_x)
	train_accuracy = accuracy_score(train_y, predict)

	# check against validation set
	predict = clf.predict(test_x)
	test_accuracy = accuracy_score(test_y, predict)

	print "Train/test accuracy:"
	print round(train_accuracy,n_round), round(test_accuracy, n_round)     
	
def boosted():
	# Define and run tree classifier
	clf = ensemble.GradientBoostingClassifier( n_estimators = 100, min_samples_leaf = min_leaf_samples, criterion = criterion) #max_leaf_nodes  loss = 'exponential',
	clf = clf.fit(train_x, train_y)
	predict = clf.predict(train_x)
	train_accuracy = accuracy_score(train_y, predict)

	# check against validation set
	predict = clf.predict(test_x)
	test_accuracy = accuracy_score(test_y, predict)

	print "Train/test accuracy:"
	print round(train_accuracy,n_round), round(test_accuracy, n_round)   
	 
def ada_boosted():
	# Define and run tree classifier
	clf = ensemble.AdaBoostClassifier( n_estimators=100 )
	clf = clf.fit(train_x, train_y)
	predict = clf.predict(train_x)
	train_accuracy = accuracy_score(train_y, predict)

	# check against validation set
	predict = clf.predict(test_x)
	test_accuracy = accuracy_score(test_y, predict)

	print "Train/test accuracy:"
	print round(train_accuracy,n_round), round(test_accuracy, n_round)   
	 
n_round = 4


# MAIN
# Get data but only keep certified and denied outcomes 
# (future work could possibly include the merger of the other outcomes, i.e. certified expired as certified)
data = pd.read_csv("data/us_perm_visas.csv", nrows = 40000,low_memory = False) #
data = data[data['case_status'].isin(['Certified', 'Denied'])]

hr_to_year = 2080

#~ data =  data[['case_status','us_economic_sector','class_of_admission','wage_offer_from_9089']].dropna()
drop_cols = ['pw_soc_code', 'case_no','pw_level_9089','naics_2007_us_code','wage_offer_from_9089', 'employer_address_1', "pw_job_title_9089", "wage_offer_unit_of_pay_9089"]
data.drop(drop_cols, axis = 1, inplace = True)
cols = data.columns

#~ print len(data[data['case_status'] == 'Denied'] )/(1.*len(data[data['case_status'] == 'Certified'] ))
# Aggresive removal of missing data:
# if a col is nan for >90% of the data then drop it so dropping
# rows with *any* nans doesn't wipe out all the data. It is insufficient
# to do just with len(data with dropped) !=0, still kills everything 
non_na_cols = []
for col in cols:
	if len(data[col].dropna()) >  0.90*len(data[col]):
		non_na_cols.append(col)

# Keep only the columns where >90% of the data are not nan, then drop the rows
# which have *ANY* entry as nan
prev = len(data)
data = data[non_na_cols].dropna()
print("Fraction of denied - we need accuracy to be much better than this or it's just a single outcome guess!")
print(round((1.-len(data[data['case_status'] == 'Denied'] )/(1.*len(data[data['case_status'] == 'Certified'] ))),n_round))

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
train_x, test_x, train_y, test_y = train_test_split(one_hot_data, out,test_size = 0.25)
ntrain = len(train_x)
print("\n")
print("--Single tree classifier--")
######### SINGLE TREE CLASSIFIER
single_tree()
print("\n")

#~ print("--Forest classifier--")
#~ ######### RANDOM FOREST TREE CLASSIFIER
#~ forest()

print("\n")
print("--Gradient boosted classifier--")
######### Boosted tree
boosted()


print("\n")
print("--Ada boosted classifier--")
######### Boosted tree
ada_boosted()
