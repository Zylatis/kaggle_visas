import pandas as pd
from sklearn import tree, linear_model, ensemble
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy.stats import randint as randint
from dat import *

def search_summary( results ):
	summary = pd.DataFrame(results['params'])	
	summary['score'] =  results['mean_train_score']
	summary.set_index('score', inplace = True)
	summary.sort_index(inplace = True, ascending = False)
	print(" Summary of parameter search:" )
	print summary.head(3)

	best = pd.Series.to_dict(summary.iloc[0])
	return best


def convert( pair ):
	a = pair[0]
	if type(pair[1]) == str:
		b = float(pair[1].replace(',', ''))
	else:
		b = pair[1]
		
	return b*salary_conversion[a]
	
# takes in classifier model, trains, gives accuracy
# classifier specific calcs need to be done outside this, but the model should be returned
# (pass by model yay)
def fit_model( clf, data_dict, param_space ):
	train_x = data_dict['train_x']
	test_x = data_dict['test_x']
	train_y = data_dict['train_y']
	test_y = data_dict['test_y']

	
	# add in stochastic hyperparam optimisation

	random_opt = GridSearchCV( clf, param_grid = param_space,  cv = 5, n_jobs = -1, return_train_score = True)
	random_opt.fit(train_x, train_y)
	#~ print random_opt.cv_results_.keys()
	best = search_summary( random_opt.cv_results_)
	
	# not sure if this is needed but we now get the model with those hyper parameters and test against the
	# test data (or if CV is done above do I feed in whole dataset?)
	clf.set_params( **best)
	clf.fit(train_x, train_y)
	train_predict = clf.predict(train_x)
	train_accuracy = accuracy_score(train_y, train_predict)

	# check against validation set
	test_predict = clf.predict(test_x)
	test_accuracy = accuracy_score(test_y, test_predict)
	print("-------------------")
	print "Accuracy of top ranked parameters:"
	print "Train: ", round(train_accuracy,n_round), " Test: ", round(test_accuracy, n_round) 
	

def single_tree(data_dict, min_leaf_samples, max_leaf_nodes, max_depth, weights = None  ):
	# Define and run tree classifier
	
	clf = tree.DecisionTreeClassifier()
	param_space = {
		'max_depth' : list(range(1,20)),
		'criterion' : ["gini", "entropy" ],
		'class_weight' : [None, 'balanced'],
		'min_samples_leaf' : [10,100],
		'max_leaf_nodes' : [ 100,1000 ]
	}
	fit_model(clf, data_dict, param_space)
	

def forest( data_dict ):
	# Define and run tree classifier
	param_space = {
		'max_depth' : list(range(1,20)),
		'criterion' : ["gini", "entropy" ],
		'class_weight' : [None, 'balanced'],
		'min_samples_leaf' : [10,100],
		'max_leaf_nodes' : [ 100,1000],
		'n_estimators' : [ 10,50 ]
	}
	clf = ensemble.RandomForestClassifier()
	fit_model(clf, data_dict, param_space)   
	
#~ def boosted( data_dict, min_leaf_samples, max_leaf_nodes, max_depth, n_estimators  ):
	#~ # Define and run tree classifier
	#~ clf = ensemble.GradientBoostingClassifier( n_estimators = n_estimators, min_samples_leaf = min_leaf_samples, max_depth = max_depth, max_leaf_nodes = max_leaf_nodes) #max_leaf_nodes  loss = 'exponential',
	#~ fit_model(clf, data_dict)

def logit( data_dict ):
	param_space = {
		'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'class_weight' : [None, 'balanced'],
		'max_iter' : [10,100,500],
		'random_state' : [0],
		'n_jobs' : [-1]
	}
	clf = linear_model.LogisticRegression()
	fit_model(clf, data_dict)
	
def NN( data_dict ):
	clf = MLPClassifier()
	param_space = {
		'solver' : ['lbfgs', 'sgd', 'adam'],
		'activation' : ['identity', 'logistic', 'tanh', 'relu'],
		#~ 'learning_rate' : ['constant', 'invscaling', 'adaptive'],
		'random_state' : [0],
		'max_iter' : [500],
		#~ 'n_jobs' : [-1],
		'hidden_layer_sizes' : [
						[2,2,2 ],
						[5,2 ],
						[10,2],
						[10,5]
						]
	}
	fit_model(clf, data_dict, param_space)
