import pandas as pd
import fns
from scipy.stats import pearsonr
import scipy.stats as ss
import numpy as np
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
    r,k = confusion_matrix.shape
    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / ( min( (kcorr-1), (rcorr-1))) )


print "##Getting data:##"
data = pd.read_csv("data/us_perm_visas.csv",  nrows = 100000, low_memory = False)
print data['case_status'].value_counts()
data['case_status'] = data['case_status'].str.replace( "Certified-Expired","Certified")
print("")
# toggle if to replace withdrawn with denied
#~ data['case_status'] = data['case_status'].str.replace( "Withdrawn","Denied")
data = data[data['case_status'].isin(['Certified', 'Denied'])]


# fix some formating errors
data['country_of_citizenship'].fillna(data['country_of_citzenship'], inplace = True)
data['decision_date']  = pd.to_datetime(data['decision_date'],format = '%Y-%m-%d')
data['decision_year'] =  data['decision_date'].apply(lambda x: x.year)
data['employer_state'] = data['employer_state'].map(fns.us_state_abbrev_cap)

pd.Series(data.columns).to_csv("all_cols.csv")
cols = data.columns
non_na_cols = []
for col in cols:
	if len(data[col].dropna()) >  0.9*len(data[col]):
		non_na_cols.append(col)
		#~ print col



# Keep only the columns where >90% of the data are not nan, then drop the rows
# which have *ANY* entry as nan
data = data[non_na_cols].dropna()
# see how many and which survived
#~ print len(data.columns)

# convert prevailing wage to salary (some are p/a some are p/hr)
temp =  [list(a) for a in zip(data['pw_unit_of_pay_9089'] , data['pw_amount_9089'])]
data['annual_salary'] = map(fns.convert, temp)


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
#~ "pw_job_title_9089",
"decision_date"] #"wage_offer_unit_of_pay_9089"	


# This is to ensure we only drop cols that are actually there!
# This process may depend on how much data we read in, and how we treat that 90% prune
drop_cols = list(set(drop_cols) & set(data.columns))
data.drop(drop_cols, axis = 1, inplace = True)
columns = data.columns
print "##Cols being used:##"
for col in data.columns:
	#~ print col
	if type(data[col][0]) == str:
		data[col] = data[col].str.upper()
print ""

data['employer_name'] = pd.Series(data['employer_name']).str.replace(".", '').str.replace(",", '').str.replace(" ", '')
data['pw_soc_title'] = pd.Series(data['pw_soc_title']).str.replace(".", '').str.replace(",", '').str.replace(" ", '')
pd.Series(data['employer_name'].value_counts()).to_csv("companies.csv")
data.to_csv("pruned_data_eda.csv")

one_hot_cat = []
label_cat = []
# See how we're doing for categoricals
print "Check categoricals"
print "----"

for col in columns:
	if type(data[col][0]) == str:
		bit = list(set(data[col]))
		print( col, len(bit) )
		if(len(bit) > 5 ):
			label_cat.append(col)
		else:
			one_hot_cat.append( col )

n_features = len(data.columns)
corr_matt = []
for col1 in data.columns:
	row = []
	for col2 in data.columns:
		confusion_matrix = pd.crosstab(data[col1].values, data[col2].values)
		row.append( cramers_corrected_stat(confusion_matrix))
	corr_matt.append(row)

matplotlib.rcParams.update({'font.size': 6})
fig, ax = plt.subplots()
im = ax.imshow(corr_matt)

# We want to show all ticks...
ax.set_xticks(np.arange(n_features))
ax.set_yticks(np.arange(n_features))
# ... and label them with the respective list entries
ax.set_xticklabels(data.columns, fontsize = 9)
ax.set_yticklabels(data.columns,fontsize = 9)
#~ ax.ticklabel_format(style='plain')
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(n_features):
    for j in range(n_features):
        text = ax.text(j, i, round(corr_matt[i][j],2),
                       ha="center", va="center", color="w")


fig.tight_layout(rect=[0, 0.00, 1, .9])

plt.title("Cramer's correlation of remaining features", fontdict = {'fontsize':15,'weight': 'bold'})

plt.savefig("corrs.png",dpi = 300)
fig.clf()
cert_income = data[data['case_status']=='CERTIFIED']['annual_salary']/1000.
den_income = data[data['case_status']=='DENIED']['annual_salary']/1000.
n, bins, patches = plt.hist(x=cert_income, bins='auto', color='red', alpha=0.7, rwidth=0.85)
n, bins, patches = plt.hist(x=den_income, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.xlim(right = 200)
plt.xlim(left = 0)
#~ plt.show()
plt.title("Salary split for certified (red) and denied (blue)", fontdict = {'fontsize':15,'weight': 'bold'})
plt.xlabel("Salary (thousands)",fontsize = 11)
plt.ylabel("Number of cases",fontsize = 11)
plt.savefig("salary.png",dpi = 300)

print cert_income.describe()
print den_income.describe()
