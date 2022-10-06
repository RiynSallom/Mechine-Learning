# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:47:41 2022

@author: ACER
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


dg=load_iris()
print(dir(dg))
print(dg.data[0])
plt.gray()
#for i in range(10):
 #   plt.matshow(dg.images[i])
print(dg.target[0:10])
print(dg.target_names[0:3])
print(dg.feature_names[0:6])
#print(dg.frame[0])
X_train, X_test, y_train, y_test =train_test_split(dg.data,dg.target,test_size=0.2)
print(len(X_train))
print(len(X_test))
model=LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test, y_test))
print(model.predict([dg.data[8]]))
y=model.predict(X_test)
c=confusion_matrix(y_test, y)
print(c)
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(c, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')