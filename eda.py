import pandas as pd
import fns
from scipy.stats import pearsonr
import dat
import numpy as np
import copy
import matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import cv2
import sys

# output folder for things
csv_output = "truncated_data/"
# Flag to determine if we want to hear the code whinge about dropping cols that
# don't exist - some cols may appear or not depending on how much data we read in
drop_errors = 'ignore'

# Import full dataset
try:
    nrows=int(input("Number of rows to import (-1 for all): "))
except ValueError as e:
	print(e)
	print("Input not a number")

print("\n---------- Getting data ----------")
if nrows == -1:
	data = pd.read_csv("data/us_perm_visas.csv",  low_memory = False)
else:
	data = pd.read_csv("data/us_perm_visas.csv", nrows = nrows,  low_memory = False)

nrows = len(data)
# Output number of each outcome
print(data['case_status'].value_counts())

# Replace certified-expired with certified, as we are interested in the certification process
data['case_status'] = data['case_status'].str.replace( "Certified-Expired","Certified")

# Toggle if to replace withdrawn with denied
data['case_status'] = data['case_status'].str.replace( "Withdrawn","Denied")
data = data[data['case_status'].isin(['Certified', 'Denied'])]

### Fix some formating errors ###
# Typo in column name
data['country_of_citizenship'].fillna(data['country_of_citzenship'], inplace = True)

# Some inconsistencies with use of abbrevs vs full state names
data['employer_state'] = list( data['employer_state'].map(dat.us_state_abbrev_cap) )

# Convert date string into python datetime format
# Also add in decision year as possible feature as additional granularity from full date may be overkill
# Could put this in function as is quite general, todo!
data['decision_date']  = pd.to_datetime(data['decision_date'],format = '%Y-%m-%d')
data['decision_year'] =  data['decision_date'].apply(lambda x: x.year)

data['decision_date_elapsed'] = copy.deepcopy(data['decision_date'] )
sorted_dates = data.sort_values(by = ['decision_date_elapsed'])['decision_date_elapsed']
start = sorted_dates.iloc[0]
data['decision_date_elapsed'] = data['decision_date_elapsed'].subtract( [start]*len(data) ).apply(lambda x: x.days)

# Continue to examine data, naming index
data.index.name = 'submission #'
columns = copy.deepcopy(data.columns) # we might make changes to this later, safer to have a copy

# We have a significan amount of missing data, make dataframe to look at
# Keep only the columns where >90% of the data are not nan, then drop the rows
# which have *ANY* entry as nan
max_perc_nan = 0.1
missing_summary = pd.DataFrame(index = columns, columns = ['% NAN'])
missing_summary.index.name = 'FEATURE'
missing_summary['% NAN'] = [ round(1.-len(data[col].dropna())/(1.*nrows),2) for col in columns]
missing_summary = missing_summary[missing_summary['% NAN'] < max_perc_nan]
non_na_columns = missing_summary.index.values

assert len(non_na_columns) > 0
data = data[non_na_columns].dropna()

# convert prevailing wage to salary (some are p/a some are p/hr)
assert 'pw_unit_of_pay_9089' in data.columns
temp =  [list(a) for a in zip(data['pw_unit_of_pay_9089'] , data['pw_amount_9089'])]
data['annual_salary'] = list( map(fns.convert, temp) )

# Manual dropping of features we think aren't informative/too fine grained/double counting
drop_columns = [
'employer_postal_code',				 # as above with postcode
'country_of_citzenship', 			 # Mis-spelled column
'pw_soc_code',						 # 
'case_no',							 # Unique for each case
'employer_address_1',				 # Likely unique with employer
]		
data.drop(drop_columns, axis = 1, inplace = True, errors = drop_errors)

# Investigate our categoricals and how best to process them
# If we have just a few different values of a given categorical feature, one hot is okay
# If not, this will take too much memory so we resort to just using label encoding

print("\n---------- Check categoricals ----------")
divided_features = fns.divide_features(data)
categoricals = copy.deepcopy(data[divided_features['one_hot'] + divided_features['label']]) #need copy here as we will drop stuff
ordinals = data[divided_features['cont']]

# For now we will just use cramers to compare categoricals and look at ordinals separately
n_ordinal = len(ordinals.columns)
n_cat = len(categoricals.columns)

print("\n")	
print("Number of categoricals: " + str(n_cat) + ". Number of ordinals: " + str( n_ordinal ) + "\n")

# Standardise all of the categoricals (all upper case, remove all punctuation and other cosmetic things to ensure we don't have duplicates)
fns.standardise_strings( categoricals )

# Collect all the cleaned up bits back into a single dataframe and poop out for later fitting
data = pd.concat([categoricals, ordinals], axis=1)
data.to_csv(csv_output+"pruned_data_eda.csv")

print("\n---------- Do plots ----------")

print("Stacked bar graphs for 'one-hot'-esque cols")
for col in divided_features['one_hot']:
	fns.stacked_bar(data,col)

print("\nCompute Cramers correlations for all, trimmed, and final set of categoricals:")
fns.cramers_corr_plot( categoricals, "full_correlation")
categoricals.drop([
	"employer_city",
	"employer_state"
	], axis = 1, inplace = True, errors = drop_errors)
fns.cramers_corr_plot( categoricals, "truncated_correlation")

print("\nPrepare ordinal plots:")
fns.plot_ordinal( data, 'annual_salary', 1000., 300)
fns.plot_ordinal( data, 'decision_date_elapsed', 1., 200.)

# Collect all the data together after trimming and so on
print("\n---------- Dumping truncated data to file ----------")


	