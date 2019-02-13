import pandas as pd
import scipy.stats as ss
from sklearn import tree, linear_model, ensemble
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy.stats import randint as randint
import dat
import copy
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    chi2 = [] # Clear stuff to save memory (as below)
    r,k = confusion_matrix.shape
    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    phi2 = []
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / ( min( (kcorr-1), (rcorr-1))) )


def get_cramers_corr( df ):
    corr_matt = []
    n_iterations = 1.*len(df.columns)**2
    count = 0.
    for col1 in df.columns:
    	row = []
    	for col2 in df.columns:
    		if col1 == col2:
    			row.append(1.)
    		else:
	    		confusion_matrix = pd.crosstab(df[col1].values, df[col2].values)
	    		row.append( cramers_corrected_stat(confusion_matrix))
    		count += 1
    	corr_matt.append(row)
    return corr_matt



# Possibly more useful to wrap all this shit in a class but fine for now
def search_summary( results, model ):
	summary = pd.DataFrame(results['params'])	
	summary['score'] =  results['mean_train_score']
	summary.set_index('score', inplace = True)
	summary.sort_index(inplace = True, ascending = False)
	print(" Summary of parameter search:" )
	# Show the top 3 sets of hyperparameters
	print(summary.head(3)) 
	summary.to_csv("models/" + model + ".csv")
	best = pd.Series.to_dict(summary.iloc[0])
	return(best)


def convert( pair ):
	a = pair[0]
	if type(pair[1]) == str:
		b = float(pair[1].replace(',', ''))
	else:
		b = pair[1]
		
	return b*dat.salary_conversion[a]
	
# takes in classifier model, trains, gives accuracy
# classifier specific calcs need to be done outside this, but the model should be returned
# (pass by model yay)
def fit_model( clf, data_dict, param_space, model ):
	train_x = data_dict['train_x']
	test_x = data_dict['test_x']
	train_y = data_dict['train_y']
	test_y = data_dict['test_y']

	
	# add in stochastic hyperparam optimisation

	random_opt = GridSearchCV( clf, param_grid = param_space,  cv = 5, n_jobs = -1, return_train_score = True)
	random_opt.fit(train_x, train_y)
	#~ print random_opt.cv_results_.keys()
	best = search_summary( random_opt.cv_results_, model)
	
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
	print("Accuracy of top ranked parameters:")
	print("Train: ", round(train_accuracy,dat.n_round), " Test: ", round(test_accuracy, dat.n_round) ) 
	

def single_tree(data_dict):
	# Define and run tree classifier
	
	clf = tree.DecisionTreeClassifier()
	param_space = {
		'max_depth' : list(range(1,20)),
		'criterion' : ["gini", "entropy" ],
		'class_weight' : [None, 'balanced'],
		'min_samples_leaf' : [10,100],
		'max_leaf_nodes' : [ 100,1000 ]
	}
	fit_model(clf, data_dict, param_space, "single_tree")
	

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
	fit_model(clf, data_dict, param_space, "forest")   
	
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
	fit_model(clf, data_dict, param_space, "logit")
	
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
	fit_model(clf, data_dict, param_space, "NN")

def divide_features( data ):
	label_cat = []
	one_hot_cat = []
	continuous = []
	for col in data.columns:
		# Get list of values
		bit = list(set(data[col]))
		# Print number of different values of this feature
		print( [ col, len(bit) ])

		# Only looking at cases where the value type is a string
		if type(data[col][0]) == str:
			bit = list(set(data[col]))
			
			# If there are more than n = 5 values for given feature, make it label encoded
			if(len(bit) > 5 ):
				label_cat.append(col)
			# if  <=5 then we can reasonably one-hot encode it using
			else:
				one_hot_cat.append( col )

		# If it's not a string, leave it alone, it's already continuous
		else:
			continuous.append(col)

	return {'one_hot' : one_hot_cat, 'label' : label_cat, 'cont' : continuous }


# At one point was going to use this to do all correlations at once using label, onehot, and leavig it alone.
# However, Cramers recovers correlation between city and state which this didn't (even full one-hot, i think, recheck)
# So, given the fact we only have two numeric features we will just consider the correlations amongst categoricals
# and look at the rest separately
def mixed_encode(data, label_encode_all = False):
	# If label_encode_all = True then apply_one_hot ignored (see below for commentary on why this is dumb and should be fixed)
	encoded_dat = copy.deepcopy(data)
	one_hot_cat = []

	if not label_encode_all:
		label_cat = []
		for col in encoded_dat.columns:
			# Get list of values
			bit = list(set(data[col]))
			# Print number of different values of this feature
			print( col, len(bit) )

			# Only looking at cases where the value type is a string
			if type(data[col][0]) == str:
				bit = list(set(data[col]))
				
				# If there are more than n = 5 values for given feature, make it label encoded
				if(len(bit) > 5 ):
					label_cat.append(col)
				# if  <=5 then we can reasonably one-hot encode it using
				else:
					one_hot_cat.append( col )
	else:
		label_cat = data.columns

	for col in label_cat:
		enc = LabelEncoder()
		enc.fit( encoded_dat[col])
		encoded_dat[col] = enc.transform(encoded_dat[col])

	# Seems weird to have allow_one_hot and include catagoricals
	# The purpose is to make this general purpose.
	# When fitting we want to do the mixed encoding using one hot for certain catagoricals and label encoding for the rest

	# For correlation matrix, however, we want to leave the catagoricals alone in this function and do something else with them, 
	# namely the Cramers correlation. For now it's easier to not have that toggle here.
	# Possibly better to have the calcualtion of cols done externally and then just have this encoder take in those cols and apply
	# Add to todo list!

	encoded_dat = pd.get_dummies(encoded_dat, columns=one_hot_cat, drop_first=True)

	return encoded_dat



def cramers_corr_plot( categoricals, filename ):
	cols = categoricals.columns
	n_features = len( cols )

	# reorder cols alphabetically to make correlations of similar stuff easier to read
	# (assumes of conceptually similar things are alphabetically similar, but reasonable if prefixes used)
	cols_sorted = sorted(cols)
	categoricals = categoricals[cols_sorted]
	cramers_corr_matrix = get_cramers_corr( categoricals )
	corr_matt = np.asarray(cramers_corr_matrix)
	matplotlib.rcParams.update({'font.size': 6})
	fig, ax = plt.subplots()
	im = ax.imshow(corr_matt)


	# We want to show all ticks
	ax.set_xticks(np.arange(n_features))
	ax.set_yticks(np.arange(n_features))
		
	ax.set_xticklabels(categoricals, fontsize = 7)
	ax.set_yticklabels(categoricals, fontsize = 7)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(n_features):
	    for j in range(n_features):
	        text = ax.text(j, i, round(corr_matt[i][j],2), ha="center", va="center", color="w")


	fig.tight_layout(rect=[0, 0.00, 	1, .9])

	plt.title("Correlation of remaining features", fontdict = {'fontsize':15,'weight': 'bold'})

	plt.savefig( filename + ".png",dpi = 400)
	fig.clf()
