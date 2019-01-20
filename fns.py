import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import inspect 
hr_to_year = 2080
# number integers to round some outputs
n_round = 4

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
def fit_model( clf, data_dict ):
	train_x = data_dict['train_x']
	test_x = data_dict['test_x']
	train_y = data_dict['train_y']
	test_y = data_dict['test_y']

	clf = clf.fit(train_x, train_y)
	train_predict = clf.predict(train_x)
	train_accuracy = accuracy_score(train_y, train_predict)

	# check against validation set
	test_predict = clf.predict(test_x)
	test_accuracy = accuracy_score(test_y, test_predict)

	print "Train/test accuracy:"
	print round(train_accuracy,n_round), round(test_accuracy, n_round) 


def single_tree(data_dict, min_leaf_samples, max_leaf_nodes ):
	# Define and run tree classifier
	clf = tree.DecisionTreeClassifier(min_samples_leaf = min_leaf_samples, max_leaf_nodes = max_leaf_nodes) #max_leaf_nodes 
	fit_model(clf, data_dict)
	print "Number of nodes in tree"
	print clf.tree_.node_count

def forest( data_dict, min_leaf_samples, max_leaf_nodes, max_depth, n_estimators ):
	# Define and run tree classifier
	clf = ensemble.RandomForestClassifier(n_jobs  = -1, n_estimators=n_estimators ,min_samples_leaf = min_leaf_samples, max_depth=max_depth,  max_leaf_nodes = max_leaf_nodes) #max_leaf_nodes 
	fit_model(clf, data_dict)   
	
def boosted( data_dict, min_leaf_samples, max_leaf_nodes, n_estimators  ):
	# Define and run tree classifier
	clf = ensemble.GradientBoostingClassifier( n_estimators = n_estimators, min_samples_leaf = min_leaf_samples, max_leaf_nodes = max_leaf_nodes ) #max_leaf_nodes  loss = 'exponential',
	fit_model(clf, data_dict)
	 
def ada_boosted( data_dict, max_depth, n_estimators  ):
	# Define and run tree classifier
	clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators )
	fit_model(clf, data_dict)

def logit( data_dict ):
	clf = linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', n_jobs = -1)
	fit_model(clf, data_dict)
