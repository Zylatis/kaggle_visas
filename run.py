import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import fns
from dat import *

# plot cumulative certified distribution with row number
# MAIN
# Get data but only keep certified and denied outcomes 
# (future work could possibly include the merger of the other outcomes, i.e. certified expired as certified)
print "##Getting data:##"
data = pd.read_csv("data/us_perm_visas.csv",nrows = 1500, low_memory = False)
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
"decision_date",
"decision_year"] #"wage_offer_unit_of_pay_9089"	

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
print "Memory usage:"
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


print("##Do encoding of catagoricals##")
for col in inp.columns:
	enc = LabelEncoder()
	enc.fit(inp[col])
	inp[col] = enc.transform(inp[col])

print("##Split + shuffle##")
train_x, test_x, train_y, test_y = train_test_split(inp, out,test_size = 0.25, random_state = 1)
ntrain = len(train_x)

print "## Scale training data using StandardScaler() ##"
# Scale to gaussian distribution, as defined *only* on train set
scaler = StandardScaler().fit(train_x)
scaled_train_x = scaler.transform(train_x)
scaled_test_x = scaler.transform(test_x)

data_dict = { 'train_x' : scaled_train_x, 'test_x' : scaled_test_x, 'train_y' : train_y, 'test_y' : test_y }

print("##Fraction of certified in each set:##")
print round( len(train_y[train_y == 1])/(1.*len(train_y)), n_round)
print round( len(test_y[test_y == 1])/(1.*len(test_y)), n_round)
print("\n")


print("-------------------Single tree classifier-------------------")
fns.single_tree(data_dict, min_leaf_samples, max_leaf_nodes, max_depth )
print("\n")

print("-------------------Forest classifier-------------------")
fns.forest( data_dict )
print("\n")

print("-------------------Logit classifier-------------------")
fns.logit(data_dict)
print("\n")

print("-------------------NN classifier-------------------")
fns.NN(data_dict)
print("\n")
