import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import inspect 

# helper function to conver to yearly wage
def convert( pair ):
	a = pair[0]
	b = pair[1]
	if a == 'yr' :
		return b
	else :
		return b*hr_to_year
		
	
# takes in classifier model, trains, gives accuracy
# classifier specific calcs need to be done outside this, but the model should be returned
# (pass by model yay)
def fit_model( clf ):
	clf = clf.fit(train_x, train_y)
	predict = clf.predict(train_x)
	train_accuracy = accuracy_score(train_y, predict)

	# check against validation set
	predict = clf.predict(test_x)
	test_accuracy = accuracy_score(test_y, predict)

	print "Train/test accuracy:"
	print round(train_accuracy,n_round), round(test_accuracy, n_round) 


def single_tree():
	# Define and run tree classifier
	clf = tree.DecisionTreeClassifier(min_samples_leaf = min_leaf_samples, criterion = criterion, max_leaf_nodes = max_leaf_nodes) #max_leaf_nodes 
	fit_model(clf)
	print "Number of nodes in tree"
	print clf.tree_.node_count

def forest():
	# Define and run tree classifier
	clf = ensemble.RandomForestClassifier(n_jobs  = -1, n_estimators=10 ,min_samples_leaf = min_leaf_samples, criterion = criterion, max_leaf_nodes = max_leaf_nodes) #max_leaf_nodes 
	fit_model(clf)     
	
def boosted():
	# Define and run tree classifier
	clf = ensemble.GradientBoostingClassifier( n_estimators = 100, min_samples_leaf = min_leaf_samples, max_leaf_nodes = max_leaf_nodes ) #max_leaf_nodes  loss = 'exponential',
	fit_model(clf)
	 
def ada_boosted():
	# Define and run tree classifier
	clf = ensemble.AdaBoostClassifier( n_estimators=100 )
	fit_model(clf) 
	 
