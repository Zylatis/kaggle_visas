import pandas as pd
import fns
from scipy.stats import pearsonr
import scipy.stats as ss
import numpy as np
import matplotlib
import copy
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Import full dataset
print "##Getting data:##"
data = pd.read_csv("data/us_perm_visas.csv", low_memory = False, nrows = 50000)
nrows = len(data)
# Output number of each outcome
print data['case_status'].value_counts()

# Replace certified-expired with certified, as we are interested in the certification process
data['case_status'] = data['case_status'].str.replace( "Certified-Expired","Certified")

# Toggle if to replace withdrawn with denied
data['case_status'] = data['case_status'].str.replace( "Withdrawn","Denied")
data = data[data['case_status'].isin(['Certified', 'Denied'])]

# Fix some formating errors #
# Typo in column name
data['country_of_citizenship'].fillna(data['country_of_citzenship'], inplace = True)

# Some inconsistencies with use of abbrevs vs full state names
data['employer_state'] = data['employer_state'].map(fns.us_state_abbrev_cap)

# Convert date string into python datetime format
# Also add in decision year as possible feature as additional granularity from full date may be overkill
data['decision_date']  = pd.to_datetime(data['decision_date'],format = '%Y-%m-%d')
data['decision_year'] =  data['decision_date'].apply(lambda x: x.year)


cols = data.columns
# We have a significan amount of missing data
# print len(data['employer_address_2'].dropna()), len(data['employer_address_2'])
missing_summary = pd.DataFrame(index = data.columns, columns = ['% NAN'])
missing_summary.index.name = 'FEATURE'
missing_summary['% NAN'] = [ round(1.-len(data[col].dropna())/(1.*nrows),2) for col in cols]
# print missing_summary['% NAN'].loc['employer_address_2']
pd.Series(data.columns).to_csv("all_cols.csv")
missing_summary = missing_summary[missing_summary['% NAN'] < 0.1]
non_na_cols = missing_summary.index.values
# Keep only the columns where >90% of the data are not nan, then drop the rows
# which have *ANY* entry as nan
data = data[non_na_cols].dropna()
# see how many and which survived
#~ print len(data.columns)

# convert prevailing wage to salary (some are p/a some are p/hr)
temp =  [list(a) for a in zip(data['pw_unit_of_pay_9089'] , data['pw_amount_9089'])]
data['annual_salary'] = map(fns.convert, temp)


# Manual dropping of features we think aren't informative/too fine grained/double counting
drop_cols = [
'naics_2007_us_title',
#~ 'pw_soc_title',
'job_info_work_state', # try to find if this is indeed redundant with employer state
'job_info_work_city', # as above with city
#~ 'pw_unit_of_pay_9089',
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
# ~ "pw_job_title_9089",
"decision_date"] #"wage_offer_unit_of_pay_9089"	


# This is to ensure we only drop cols that are actually there!
# This process may depend on how much data we read in, and how we treat that 90% prune
drop_cols = list(set(drop_cols) & set(data.columns))
data.drop(drop_cols, axis = 1, inplace = True)
columns = data.columns
for col in data.columns:
	if type(data[col][0]) == str:
		data[col] = data[col].str.upper()

# Standardise all of the remaining strings (remove all punctuation and other cosmetic things to ensure we don't have duplicates)
data['employer_name'] = pd.Series(data['employer_name']).str.replace(".", '').str.replace(",", '').str.replace(" ", '')
data['pw_soc_title'] = pd.Series(data['pw_soc_title']).str.replace(".", '').str.replace(",", '').str.replace(" ", '')
pd.Series(data['employer_name'].value_counts()).to_csv("companies.csv")
data.to_csv("pruned_data_eda.csv")


# Investigate our categoricals and how best to process them
# If we have just a few different values of a given categorical feature, one hot is okay
# If not, this will take too much memory so we resort to just using label encoding
one_hot_cat = []
label_cat = []
print "Check categoricals"
print "----"

for col in columns:
	# Only consider this if we have string values
	if type(data[col][0]) == str:
		# Get list of values
		bit = list(set(data[col]))
		# Print number of different values of this feature
		print( col, len(bit) )
		# If we have more than 5, label encode, otherwise one-hot encode
		if(len(bit) > 5 ):
			label_cat.append(col)
		else:
			one_hot_cat.append( col )

# The above process will result in a split, some label, some one-hot
# To deal with use we uses a helper function which we use on a deep copy to be safe

encoded_dat = copy.deepcopy(data)
encoded_dat = fns.mixed_encode( encoded_dat , True)

# exit(0)
# print("##Do encoding of catagoricals##")
# for col in encoded_dat.columns:
# 	enc = LabelEncoder()
# 	enc.fit(encoded_dat[col])
# 	encoded_dat[col] = enc.transform(encoded_dat[col])

# vestigial comment
# encoded_dat = pd.concat((encoded_dat[label_cat], pd.get_dummies(encoded_dat, columns=one_hot_cat, drop_first=True)),axis=1)

print("Do plots")

# Compute Pearesons correlations and plot as a heatmap
corr_matt = encoded_dat.corr().as_matrix()
matplotlib.rcParams.update({'font.size': 6})
fig, ax = plt.subplots()
im = ax.imshow(corr_matt)
n_features = len(encoded_dat.columns)

# We want to show all ticks...
ax.set_xticks(np.arange(n_features))
ax.set_yticks(np.arange(n_features))
ax.set_xticklabels(encoded_dat.columns, fontsize = 9)
ax.set_yticklabels(encoded_dat.columns,fontsize = 9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(n_features):
    for j in range(n_features):
        text = ax.text(j, i, round(corr_matt[i][j],2),
                       ha="center", va="center", color="w")


fig.tight_layout(rect=[0, 0.00, 1, .9])

plt.title("Correlation of remaining features", fontdict = {'fontsize':15,'weight': 'bold'})

plt.savefig("corrs.png",dpi = 300)
fig.clf()


# One thing to investigate is if 
cert_income = data[data['case_status']=='CERTIFIED']['annual_salary']/1000.
den_income = data[data['case_status']=='DENIED']['annual_salary']/1000.
n, bins, patches = plt.hist(x=cert_income, bins='auto', color='red', alpha=0.7, rwidth=0.85,histtype='step'	)
n, bins, patches = plt.hist(x=den_income, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85,histtype='step')
plt.xlim(right = 200)
plt.xlim(left = 0)
plt.title("Salary split for certified (red) and denied (blue)", fontdict = {'fontsize':15,'weight': 'bold'})
plt.xlabel("Salary (thousands)",fontsize = 11)
plt.ylabel("Number of cases",fontsize = 11)
plt.savefig("salary.png",dpi = 300)

print cert_income.describe()
print den_income.describe()



