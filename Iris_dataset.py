#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:08:14 2019

@author: ayazurrahman
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = pd.read_csv("iris.csv")

data.replace(to_replace=['Iris-setosa','Iris-versicolor','Iris-virginica'],
                         value = ['1','2','3'],
                         inplace = False)

iris = data.drop('Id', axis = 1)

X = iris.drop(['Species'], axis = 1)
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))