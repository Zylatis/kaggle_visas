import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import inspect 

# number integers to round some outputs
min_leaf_samples = 100 # interesting: if this is large then the tree only has 'certified' nodes which gives 85% accuracy
max_leaf_nodes = 1000
criterion = "entropy" # "gini"

n_round = 4


# MAIN
# Get data but only keep certified and denied outcomes 
# (future work could possibly include the merger of the other outcomes, i.e. certified expired as certified)
data = pd.read_csv("data/us_perm_visas.csv", nrows = 5000,low_memory = False) #
data = data[data['case_status'].isin(['Certified', 'Denied'])]
pd.Series(data.columns).to_csv("all_cols.csv")

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


pd.Series(data.columns).to_csv("used_cols.csv")
# convert prevailing wage to salary (some are p/a some are p/hr)
temp =  [list(a) for a in zip(data['pw_unit_of_pay_9089'] , data['pw_amount_9089'])]
data['annual_salary'] = map(convert, temp)

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
#~ ######### SINGLE TREE CLASSIFIER
single_tree()
#~ print("\n")

#~ print("--Forest classifier--")
#~ ######### RANDOM FOREST TREE CLASSIFIER
#~ forest()

#~ print("\n")
#~ print("--Gradient boosted classifier--")
#~ ######### Boosted tree
#~ boosted()


print("\n")
print("--Ada boosted classifier--")
######### Boosted tree
ada_boosted()
