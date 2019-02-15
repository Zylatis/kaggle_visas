# US perm. visa application dataset analysis 
	
This is a basic EDA and model fitting code looking at this Kaggle dataset

https://www.kaggle.com/jboysen/us-perm-visas/home

Very much a WIP, please don't hurt me for writing Python like a cpp person.

### Dataset
This dataset contains 370,000 records of permanent visa applications to the US. Applications can be either certified, denied, withrawn, or certified-expired. The data covers just a few years and has 154 columns. It has significant quantities of missing data and many typos in strings (some in column names making for extra columns).

### Goal
The obvious goal here is to make a model to predict the outcome of an application based on some or all of the information provided above. However, I wanted to go further than that and to make an app which would aid someone wanting to increase their chances of a succesfull application - there are aspects one cannot control (country of citizenship, skillset, etc) and some that one can (employer, location, salary) though each to varying degrees. My hope was that one could use a model of this data to allow users to optimimze the parameters they are willing to compromise on, say, location, in order to maximise their chances of getting certified. 

For this reason I have avoided using dimensionality reducing methods such as PCA as I wish to retain the meaningfull-ness (yes, a word, maybe) of these features and not mash them together too much.


### Prelim results
Unfortuntely, this looks to be very tricky.

#### EDA
Firstly, the data is massively unbalanced with 95% of the cases being certified. This changes slightly dedpending on how one chooses to deal with the withdrawn and certified-expired cases, but not by much. As noted previously there are large amounts of missing data. By requiring that we only keep features for which we have at least 90% of the data not NaN and dropping the rest we find we are left with only 15 or so features to work with. Almost all are categorical but one can engineer a few additional ordinal ones by converting the decision date to decision year and/or turning this into a 'days elapsed' from t = 0 kind of thing. 

Once we are left with this subset of features we can look at the correlation between them all to see if some are redundant. A few clearly are by inspection: case number is going to be unique and is useless, employer address and employer name are likely to be very closely correlated, and some other features are just index codes to reference other features (e.g. PW_SOC_CODE and PW_SOC_TITLE). In order to do the correlation calculation on the remaining features we look to Cramer's correlation function, the code for which I shamelessly pinched from some poor unsuspecting SO post. This is essentially a chi-sq test to look at correlation between categoricals. This involves an effective one-hot encoding step and so can be a bit of a memory hog. Nevertheless, it far outperforms simply using label encoding and Pearsons correlation - for one, the latter doesn't pick up correlation between states and cities, a pretty big red flag.

Overall no major correlations are found between the features, including, unfortunately, between features and the visa outcome (certified/denied). Nevertheless, this is but one metric, we can attempt a model anyway

#### Trees, trees erry'where
To be written - short answer is we can't do better than just labeling everything as certified, even if we take into account balancing and try SVM, NN, Forests, etc.
