from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import datasets
dataset = datasets.load_breast_cancer()
svm = LinearSVC()
# create the RFE model for the svm classifier
# and select attributes
rfe = RFE(svm, 5)
rfe = rfe.fit(dataset.data, dataset.target)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)










