import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
data = pd.read_csv("data/us_perm_visas.csv",nrows=150000, low_memory = False)
cols = data.columns
data = data[data['case_status'].isin(['Certified', 'Denied'])]

data =  data[['case_status','us_economic_sector','class_of_admission','wage_offer_from_9089']].dropna()


inp = data.drop('case_status', axis = 1)
out = data['case_status']
#~ certified = data[data['case_status'] == 'Certified']
#~ print certified.groupby('class_of_admission').count()
#~ plt.bar(data['case_status'], data['class_of_admission'], 1)
#~ plt.show()


one_hot_data = pd.get_dummies(inp,drop_first=True)


n = len(one_hot_data)
ntrain = int(round(0.9*n))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(one_hot_data[:ntrain], out[:ntrain])

predict = clf.predict(one_hot_data[:ntrain])
check = list(zip(out[:ntrain].values, predict))

count = 0
for el in check:
	if el[0] == el[1]:
		count = count+1

print (1.*count)/(1.*ntrain)


predict = clf.predict(one_hot_data[ntrain+1:n])
check = list(zip(out[ntrain+1:n].values, predict))

count = 0
for el in check:
	if el[0] == el[1]:
		count = count+1

print (1.*count)/(1.*(n-ntrain))

#~ dot_data = tree.export_graphviz(clf, out_file=None, feature_names=one_hot_data.columns, class_names=out, filled=True, rounded=True,  special_characters=True) 
#~ graph = graphviz.Source(dot_data) 
#~ graph.render("iris")
