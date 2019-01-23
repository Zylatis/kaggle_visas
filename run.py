import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
import fns


# number integers to round some outputs
n_round = 4
# plot cumulative certified distribution with row number
# MAIN
# Get data but only keep certified and denied outcomes 
# (future work could possibly include the merger of the other outcomes, i.e. certified expired as certified)
print "##Getting data:##"
data = pd.read_csv("data/us_perm_visas.csv", low_memory = False, nrows = 1500)
print data['case_status'].value_counts()
data['case_status'] = data['case_status'].str.replace( "Certified-Expired","Certified")
print "MEM:"
print data.memory_usage(deep=True).sum()/(10.**9)
#~ print data.dtypes
print("")
# toggle if to replace withdrawn with denied
#~ data['case_status'] = data['case_status'].str.replace( "Withdrawn","Denied")
data = data[data['case_status'].isin(['Certified', 'Denied'])]
temp = list(set(data['case_status']))
data['country_of_citizenship'].fillna(data['country_of_citzenship'], inplace = True)

data['decision_date_elapsed']  = pd.to_datetime(data['decision_date'],format = '%Y-%m-%d')
data['decision_year'] =  data['decision_date_elapsed'].apply(lambda x: x.year)
sorted_dates = data.sort_values(by = ['decision_date_elapsed'])['decision_date_elapsed']
start = sorted_dates.iloc[0]

data['decision_date_elapsed'] = data['decision_date_elapsed'].subtract( [start]*len(data) ).apply(lambda x: x.days)
data['employer_state'] = data['employer_state'].map(fns.us_state_abbrev_cap)

pd.Series(data.columns).to_csv("all_cols.csv")
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
print len(cols), len(data.columns), "\n"
# convert prevailing wage to salary (some are p/a some are p/hr)
temp =  [list(a) for a in zip(data['pw_unit_of_pay_9089'] , data['pw_amount_9089'])]
data['annual_salary'] = map(fns.convert, temp)

# add global info (hot tip from kaggle comment) regarding country success rate
# (is it worth replacing country with this alltogether?)
#~ countries = list(set(data['country_of_citizenship']))
#~ country_cert_rate = {}
#~ for country in countries:
	#~ country_data = data[data['country_of_citizenship'] == country ]['case_status']
	#~ n_cert = len(country_data[country_data == 1.])
	#~ country_cert_rate[country] = n_cert/(1.* len(country_data))
#~ data['country_cert_rate'] = data['country_of_citizenship'].map(country_cert_rate)

drop_cols = [
'naics_2007_us_title',
'pw_soc_title',
'job_info_work_state', # try to find if this is indeed redundant with employer state
'job_info_work_city', # as above with city
'pw_unit_of_pay_9089',
'pw_amount_9089',
'wage_offer_unit_of_pay_9089',
'employer_postal_code',
'country_of_citzenship', 
'pw_soc_code',
'case_no',
'pw_level_9089',
'naics_2007_us_code',
'wage_offer_from_9089',
'employer_address_1',
"pw_job_title_9089",
"decision_date"] #"wage_offer_unit_of_pay_9089"	

print data['case_status'].value_counts()

# This is to ensure we only drop cols that are actually there!
# This process may depend on how much data we read in, and how we treat that 90% prune
drop_cols = list(set(drop_cols) & set(data.columns))
data.drop(drop_cols, axis = 1, inplace = True)
columns = data.columns
print "##Cols being used:##"
for col in data.columns:
	print col
	if type(data[col][0]) == str:
		data[col] = data[col].str.upper()
print ""


print "MEM:"
print data.memory_usage(deep=True).sum()/(10.**9)

data['employer_name'] = pd.Series(data['employer_name']).str.replace(".", '').str.replace(",", '').str.replace(" ", '')
pd.Series(data['employer_name'].value_counts()).to_csv("companies.csv")
data.to_csv("pruned_data.csv")
#~ exit(0)
# See how we're doing for categoricals
for col in columns:
	if type(data[col][0]) == str:
		bit = list(set(data[col]))
		print( col, len(bit) )
		
print("---")
#~ exit(0)
# drop the target variable from training set
inp = data.drop('case_status',axis = 1) 
pd.Series(data.columns).to_csv("used_cols.csv")

# define target variable
print "##Re-jig target variables##"
data.loc[data.case_status == 'CERTIFIED', 'case_status'] = 1.
data.loc[data.case_status == 'DENIED', 'case_status'] = 0.
out = data['case_status'].astype('float')

#~ print inp['application_type'].value_counts()
#~ print inp['class_of_admission'].value_counts()
#~ print inp['country_of_citizenship'].value_counts()

#~ print inp['employer_city'].value_counts().sort_index()
#~ print inp['employer_name'].value_counts().sort_index()
#~ print inp['employer_state'].value_counts().sort_index()

#~ print inp['pw_source_name_9089'].value_counts().sort_index()
#~ print inp['us_economic_sector'].value_counts().sort_index()
#~ print data['pw_soc_code'].value_counts()

#~ exit(0)

print "##Do one-hot encoding of catagoricals##"
# Do one-hot transformation on the categorical data. 
# This is super cool and basically a way to make sure we have binary values for the categories
# rather than label by number, which will bugger up the data/scaling
#~ one_hot_data = sparse.csr_matrix( pd.get_dummies(inp,drop_first=True).values )

#~ oh_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
#~ label_enc = preprocessing.LabelEncoder()

#~ oh_data = oh_enc.fit_transform(inp)
#~ label_data = label_enc.fit_transform(inp)


for col in inp.columns:
	enc = LabelEncoder()
	enc.fit(inp[col])
	inp[col] = enc.transform(inp[col])

#~ exit(0)
print "##Split + shuffle##"

train_x, test_x, train_y, test_y = train_test_split(inp, out,test_size = 0.25, random_state = 1)

#~ train_x = train_x.todense()
#~ test_x = test_x.todense()
ntrain = len(train_x)

print "##To-dense + scale ##"
# Scale to gaussian distribution, as defined *only* on train set
scaler = preprocessing.StandardScaler().fit(train_x)
scaled_train_x = scaler.transform(train_x)
scaled_test_x = scaler.transform(test_x)

data_dict = { 'train_x' : scaled_train_x, 'test_x' : scaled_test_x, 'train_y' : train_y, 'test_y' : test_y }

print("##Fraction of certified in each set:##")
print round( len(train_y[train_y == 1])/(1.*len(train_y)), n_round)
print round( len(test_y[test_y == 1])/(1.*len(test_y)), n_round)
#~ exit(0)

# Tree parameters
min_leaf_samples = 1 # interesting: if this is large then the tree only has 'certified' nodes which gives 85% accuracy
max_leaf_nodes = None
max_depth = 20
print("")
print("--Single tree classifier--")
######### SINGLE TREE CLASSIFIER
fns.single_tree(data_dict, min_leaf_samples, max_leaf_nodes, max_depth )
print("\n")

#~ print("--Forest classifier--100")
#~ ######### RANDOM FOREST TREE CLASSIFIER
#~ fns.forest( data_dict, min_leaf_samples, max_leaf_nodes,max_depth, 100 )

print("--Forest classifier--500")
######### RANDOM FOREST TREE CLASSIFIER
fns.forest( data_dict, min_leaf_samples, max_leaf_nodes,max_depth, 500 )


print("")
print("--Gradient boosted classifier--")
######### Boosted tree
fns.boosted(data_dict, min_leaf_samples, max_leaf_nodes, max_depth, 100 )


print("")
print("--Ada boosted classifier--")
######### Boosted tree
fns.ada_boosted(data_dict, max_depth, 50)

print("")
print("--logit--")
######### Boosted tree
fns.logit(data_dict)
