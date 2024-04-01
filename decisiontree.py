import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def cal_decisionTree(file_train, file_test):
  dftrn = pd.read_csv(file_train, header=None)
  dftst = pd.read_csv(file_test, header=None)
  row_count, column_count = dftrn.shape
  cc = column_count -1
  x_train = dftrn.iloc[:,:-1]
  y_train = dftrn.iloc[:,cc]
  x_test = dftst.iloc[:,:-1]
  y_test = dftst.iloc[:,cc]
  clf = DecisionTreeClassifier()
  clf = clf.fit(x_train, y_train)
  predictions = clf.predict(x_test)
  ascore = accuracy_score(y_test, predictions)*100
  print("---------------------------Decision Tree result----------------------------")
  print("-------------------------------nhutb2014938--------------------------------")
  print("-Accuracy Score:", ascore, "%")
  print("-Confusion Matrix: \n", metrics.confusion_matrix(y_test, predictions))
  print("---------------------------------------------------------------------------")

cal_decisionTree("data\\fp\\fp.trn", "data\\fp\\fp.tst")