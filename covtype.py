import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("covtype.csv")

X = data.iloc[:,:54]  # all rows, all the features and no labels
y = data.iloc[:, 54]  # all rows, label only

# X=preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

model_tree = DecisionTreeClassifier(criterion="entropy", class_weight="balanced")
model_tree.fit(X_train, y_train)

y_pred = model_tree.predict(X_test)

print model_tree.score(X_test, y_test)