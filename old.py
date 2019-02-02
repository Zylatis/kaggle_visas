
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




corr_matt = []
for col1 in data.columns:
	row = []
	for col2 in data.columns:
		confusion_matrix = pd.crosstab(data[col1].values, data[col2].values)
		row.append( cramers_corrected_stat(confusion_matrix))
	corr_matt.append(row)
