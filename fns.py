import pandas as pd
import scipy.stats as ss
from sklearn import tree, linear_model, ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats import randint as randint
import dat
import sys
import copy
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
import multiprocessing as mp

# Hacky but functional!
# This is because of some stuff with the Google Cloud install
# (should know more about this, add it to the pile!)
try:
	f = open("matplotlib_config.dat", "r")
except:
	print("Couldn't find matplotlib_config.dat")
	exit(0)
config = f.read().strip()
assert len(config) != 0
if config != 'Agg' and config != 'pass':
	print("Invalid matplotlib config")
	exit(0)
if config == 'Agg':		
	matplotlib.use('Agg')

import matplotlib.pyplot as plt
img_folder = "imgs/"
eps = 10**(-3)


# Come back and convert this to lambda!
def clean_col( string ):
	return filter(str.isalnum, string).upper()

def standardise_strings( df_slice ):
	# rubbish = [ ".", ",", " ", "-", ""]
	cols = df_slice.columns
	for col in cols:
		assert type(df_slice[col][0]) == str
		df_slice[col] = map(clean_col, df_slice[col])

def cramers_corrected_stat(confusion_matrix):
	# Shamelessly pinched from someone on SO which references
	# Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328
	r,k = confusion_matrix.shape
	chi2 = ss.chi2_contingency(confusion_matrix)[0]
	n = confusion_matrix.sum().sum()
	confusion_matrix = [] # nyx to save memory though not much really, need to check how scope works in detail in python
	phi2 = chi2/n
	phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
	rcorr = r - ((r-1)**2)/(n-1)
	kcorr = k - ((k-1)**2)/(n-1)
	return np.sqrt(phi2corr / ( min( (kcorr-1), (rcorr-1)) + eps)  )

def cramers_corr_inner( args ):
	df, col1 = args
	cols = df.columns
	row = []
	for col2 in cols:
		if col2 == col1:
			row.append(1.)
		else:
			confusion_matrix = pd.crosstab(df[col1].values, df[col2].values).astype('int32')
			row.append( cramers_corrected_stat(confusion_matrix))
	return row


def get_cramers_corr( df, parallel = False ):
	# Computes Cramers correlation coefficient for a dataframe consisting *only*
	# of categoricals. Uses 'cramers_corrected_stat()' for a given confusion matrix
	# for a pair of features
	corr_matt = []
	n_iterations = 1.*len(df.columns)**2
	count = 0.

	# Includes paralle imp as a flag initially to make sure the paralle imp. worked properly.
	# Also makes for a good test case, potentially
	# ALSO NOTE: this parallel code, being done with multiprocessing and thus hamstrung by the GIL,
	# will make n_core copies of the entire dataframe, one for each 'thread', so don't use if memory an issue
	# (or move to numba or something)
	if( parallel ):
		p = mp.Pool( mp.cpu_count() )
		args = [ [df, col] for col in df.columns ]
		corr_matt = p.map(cramers_corr_inner, args )
	else:
		for col1 in df.columns:
			row = []
			for col2 in df.columns:
				if col1 == col2:
					row.append(1.)
				else:
					confusion_matrix = pd.crosstab(df[col1].values, df[col2].values).astype('int32')
					row.append( cramers_corrected_stat(confusion_matrix))
					confusion_matrix = []	
				count += 1
			corr_matt.append(row)
	return corr_matt


def search_summary( results, model ):
	# Possibly more useful to wrap all this shit in a class but fine for now
	
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
	# Helper function used to convert all salaries to annual
	a = pair[0]
	if type(pair[1]) == str:
		b = float(pair[1].replace(',', ''))
	else:
		b = pair[1]
		
	return b*dat.salary_conversion[a]
	
def fit_model( clf, data_dict, param_space, model ):
	# takes in classifier model, trains, gives accuracy
	# classifier specific calcs need to be done outside this, but the model should be returned
	# (pass by model yay)

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

def divide_features( data, threshold = 5 ):
	assert threshold >= 0 
	# Divvies up features into three categories: ordinal, one hot categoricals, and label encoded categoricals
	# The ordinals are any numeric features, and the remaining categoricals are split according to the number
	# of unique values - if > threshold then label encoding is used, else one-hot
	label_cat = []
	one_hot_cat = []
	continuous = []
	feature_count_summary = []
	
	for col in data.columns:
		# Get list of values
		bit = list(set(data[col]))
		# Print number of different values of this feature
		feature_count_summary.append( [ col, len(bit) ] )
		
		# Only looking at cases where the value type is a string
		if type(data[col][0]) == str:
			bit = list(set(data[col]))
			
			# If there are more than n = threshold values for given feature, make it label encoded
			if(len(bit) > threshold ):
				label_cat.append(col)
			# if  <= threshold then we can reasonably one-hot encode it using
			else:
				one_hot_cat.append( col )

		# If it's not a string, leave it alone, it's already continuous
		else:
			continuous.append(col)

	feature_count_summary = pd.DataFrame(feature_count_summary, columns = ["Feature", "Counts"])
	print(feature_count_summary)
	print("\n")

	print("Ordinal features: " + ', '.join(continuous))
	return {'one_hot' : one_hot_cat, 'label' : label_cat, 'cont' : continuous }


def encode_features( df, feature_types ):
	for col in feature_types['label']:
		enc = LabelEncoder()
		enc.fit( df[col])
		df[col] = enc.transform(df[col])

	return pd.get_dummies(df, columns=feature_types['one_hot'], drop_first=True)
	

def cramers_corr_plot( categoricals, filename ):
	cols = categoricals.columns
	n_features = len( cols )

	# reorder cols alphabetically to make correlations of similar stuff easier to read
	# (assumes of conceptually similar things are alphabetically similar, but reasonable if prefixes used)
	cols_sorted = sorted(cols)
	categoricals = categoricals[cols_sorted]
	cramers_corr_matrix = get_cramers_corr( categoricals, parallel = True )
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


	fig.tight_layout(rect=[0, 0.00, 1, .9])

	plt.title("Cramers V correlation for categoricals", fontdict = {'fontsize':10,'weight': 'bold'})

	plt.savefig( img_folder + filename + ".png",dpi = 400)
	fig.clf()


# Plot ordinals against case outcome
def plot_ordinal( df, col, y_scale, x_max):
	print("Plotting ordinal column " + col )
	assert y_scale > 0
	assert x_max > 0
	assert col in df.columns
	
	certified_data = df[df['case_status']=='CERTIFIED'][col].values/y_scale
	denied_data = df[df['case_status']=='DENIED'][col].values/y_scale
	
	n, bins, patches = plt.hist(x=certified_data, bins='auto', color='red', alpha=0.7, histtype='step')
	H1 = [ [bins[i], n[i] ] for i in range(len(n))]
	n, bins, patches = plt.hist(x=denied_data, bins='auto', color='#0504aa', alpha=0.7,histtype='step')
	H2 = [ [bins[i], n[i] ] for i in range(len(n))]
	
	plt.xlim(right = x_max)
	plt.xlim(left = 0) # assumes we won't be plottin negative stuff for now
	plt.title(col + " split for certified (red) and denied (blue)", fontdict = {'fontsize':10,'weight': 'bold'})
	plt.xlabel(col,fontsize = 11)
	plt.ylabel("Number of cases",fontsize = 11)
	plt.savefig( img_folder + col + "_split.png",dpi = 400)
	plt.clf()

	anova = ss.f_oneway(certified_data, denied_data)
	pval = round(anova.pvalue,4)
	print("ANOVA p-val " + str(pval) + "\n")



def stacked_bar( df, col ):
	print "		" + col
	width = 1

	# want the totals to sort by height to make comparisons easier
	totals = df[col].value_counts().sort_values(ascending = False)
	vals =  list(totals.index)
	n_vals = len(vals)
	
	certified_data = df[df['case_status']=='CERTIFIED'][col].value_counts()
	denied_data = df[df['case_status']=='DENIED'][col].value_counts()

	certified_data_ordered = [ certified_data[x] if x in certified_data  else 0 for x in vals ]
	denied_data_ordered = [ denied_data[x] if x in denied_data  else 0 for x in vals ]
	
	x_space = np.arange( n_vals )
	p1 = plt.bar(x_space, certified_data_ordered , width)
	p2 = plt.bar(x_space, denied_data_ordered, width, bottom=certified_data_ordered)
	plt.xticks(x_space, vals, rotation='vertical')
	plt.legend((p1[0], p2[0]), ('Certified', 'Denied'))
	plt.title(" Certified/denied split for "+ col, fontdict = {'fontsize':10,'weight': 'bold'})
	plt.ylabel("Number of cases",fontsize = 11)
	
	plt.savefig( img_folder + col + "_split.png",dpi = 400)
	plt.clf()