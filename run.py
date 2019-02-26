import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import fns
import copy
n_round = 4
# plot cumulative certified distribution with row number
# MAIN
# Get data but only keep certified and denied outcomes 
# (future work could possibly include the merger of the other outcomes, i.e. certified expired as certified)

# Import full dataset
try:
    nrows=int(input("Number of rows to import (-1 for all): "))
except ValueError as e:
	print(e)
	print("Input not a number")

print "##Getting data:##"
file = "truncated_data/pruned_data_eda.csv"
if nrows == -1:
	data = pd.read_csv(file,  low_memory = False)
else:
	data = pd.read_csv(file, nrows = nrows,  low_memory = False)

data.set_index('submission #', inplace = True)
inp = data.drop('case_status',axis = 1) 
print("Imported " + str(len(data)) + " rows")
# define target variable
print "##Re-jig target variables##"
data.loc[data.case_status == 'CERTIFIED', 'case_status'] = 1.
data.loc[data.case_status == 'DENIED', 'case_status'] = 0.
out = data['case_status'].astype('float')

print("##Fraction of certified over data (truncated in eda.py) ##")
print round( len(data[data['case_status'] == 1])/(1.*len(data)), n_round)


print("\n---------- Check categoricals ----------")
divided_features = fns.divide_features(inp)
categoricals = copy.deepcopy(inp[divided_features['one_hot'] + divided_features['label']]) #need copy here as we will drop stuff
ordinals = inp[divided_features['cont']]

# Conver to float64 to stop shut up the scaler
inp = fns.encode_features(inp, divided_features).astype('float64')

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
train_frac =  round( len(train_y[train_y == 1])/(1.*len(train_y)), n_round)
test_frac = round( len(test_y[test_y == 1])/(1.*len(test_y)), n_round)

print(train_frac, test_frac)
print("\n")


# print("-------------------Single tree classifier-------------------")
# fns.single_tree(data_dict)
# print("\n")

# print("-------------------Forest classifier-------------------")
# fns.forest( data_dict )
# print("\n")

print("-------------------XGBoost classifier-------------------")
fns.XGBoost( data_dict, train_frac )
print("\n")


# print("-------------------Logit classifier-------------------")
# fns.logit(data_dict)
# print("\n")

# print("-------------------NN classifier-------------------")
# fns.NN(data_dict)
# print("\n")

# print("-------------------SVM classifier-------------------")
# fns.SVM(data_dict)
# print("\n")
