import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import inspect 
import fns
# number integers to round some outputs
n_round = 4

# plot cumulative certified distribution with row number
# MAIN
# Get data but only keep certified and denied outcomes 
# (future work could possibly include the merger of the other outcomes, i.e. certified expired as certified)
print "Getting data:"
data = pd.read_csv("data/us_perm_visas.csv", nrows = 100000, low_memory = False) #
data['case_status'] = data['case_status'].str.replace( "Certified-Expired","Certified")

print len(data[data['case_status']=='Certified']),len(data[data['case_status']=='Denied']), len(data[data['case_status']=='Withdrawn']), len(data)

print 'Frac certified from all data:'
data = data[data['case_status'].isin(['Certified', 'Denied'])]
print round(len(data[data['case_status'] == 'Certified'])/(1.*len(data)),n_round)
temp = list(set(data['case_status']))
data['country_of_citizenship'].fillna(data['country_of_citzenship'], inplace = True)


data['decision_date_elapsed']  = pd.to_datetime(data['decision_date'],format = '%Y-%m-%d')
data['decision_year'] =  data['decision_date_elapsed'].apply(lambda x: x.year)
sorted_dates = data.sort_values(by = ['decision_date_elapsed'])['decision_date_elapsed']
start = sorted_dates.iloc[0]

data['decision_date_elapsed'] = data['decision_date_elapsed'].subtract( [start]*len(data) ).apply(lambda x: x.days)
#~ data.drop(['employer_postal_code', 'employer_name','country_of_citzenship', 'job_info_work_state', 'decision_date','job_info_work_city'], inplace = True, axis = 1)

pd.Series(data.columns).to_csv("all_cols.csv")
#~ data =  data[['case_status','us_economic_sector','class_of_admission', 'pw_amount_9089','pw_unit_of_pay_9089','wage_offer_from_9089','country_of_citizenship']]
drop_cols = ['country_of_citzenship', 'pw_soc_code', 'case_no','pw_level_9089','naics_2007_us_code','wage_offer_from_9089', 'employer_address_1', "pw_job_title_9089","decision_date"] #"wage_offer_unit_of_pay_9089"	
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
print "\n"
# see how many and which survived
print "Old v new number of cols kept"
print len(cols), len(data.columns)

# convert prevailing wage to salary (some are p/a some are p/hr)
temp =  [list(a) for a in zip(data['pw_unit_of_pay_9089'] , data['pw_amount_9089'])]
data['annual_salary'] = map(fns.convert, temp)
print "Cols being used:"
for col in data.columns:
	print col

# drop the target variable from training set
inp = data.drop('case_status',axis = 1) 
pd.Series(data.columns).to_csv("used_cols.csv")

# define target variable
data.loc[data.case_status == 'Certified', 'case_status'] = 1
data.loc[data.case_status == 'Denied', 'case_status'] = 0
out = data['case_status']

#~ exit(0)
# Do one-hot transformation on the categorical data. 
# This is super cool and basically a way to make sure we have binary values for the categories
# rather than label by number, which will bugger up the data/scaling
one_hot_data = pd.get_dummies(inp,drop_first=True)
train_x, test_x, train_y, test_y = train_test_split(one_hot_data, out,test_size = 0.25)
ntrain = len(train_x)

data_dict = { 'train_x' : train_x, 'test_x' : test_x, 'train_y' : train_y, 'test_y' : test_y }

print("Fraction of certified in each set:")
print round( len(train_y[train_y == 1])/(1.*len(train_y)), n_round)
print round( len(test_y[test_y == 1])/(1.*len(test_y)), n_round)
#~ exit(0)

# Tree parameters
min_leaf_samples = 5 # interesting: if this is large then the tree only has 'certified' nodes which gives 85% accuracy
max_leaf_nodes = 50000
criterion = "entropy" # "gini"

print("\n")
print("--Single tree classifier--")
######### SINGLE TREE CLASSIFIER
fns.single_tree(data_dict, min_leaf_samples, max_leaf_nodes )
print("\n")

print("--Forest classifier--")
######### RANDOM FOREST TREE CLASSIFIER
fns.forest( data_dict, min_leaf_samples, max_leaf_nodes, 100 )

#~ print("\n")
#~ print("--Gradient boosted classifier--")
#~ ######### Boosted tree
#~ fns.boosted(data_dict, min_leaf_samples, max_leaf_nodes, 100 )


print("\n")
print("--Ada boosted classifier--")
######### Boosted tree
fns.ada_boosted(data_dict, 100)
