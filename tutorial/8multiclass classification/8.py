# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:57:44 2022

@author: ACER
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
df=pd.read_csv("salaries.csv")
print(df.head())
inputs=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']
print('\n',inputs)
print('\n',target)
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])
inputs_n=inputs.drop(['company','job','degree'],axis='columns')
print(inputs_n)
model=tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
print(model.score(inputs_n, target))
print(model.predict([[2,2,1],]))
print(model.predict([[2,0,1],]))


#######################EX
de=pd.read_csv('titanic.csv')
print(de.columns)
de=de.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
print(de.head())
target_ex=de.Survived
inputs_Ex=de.drop('Survived',axis='columns')
inputs_Ex.Sex=inputs_Ex.Sex.map({'male':1,'female':2})
inputs_Ex.Age=inputs_Ex.Age.fillna(inputs_Ex.Age.mean())
print(inputs_Ex.head())
X_train, X_test, y_train, y_test=train_test_split(inputs_Ex,target_ex,test_size=0.2)
model_Ex=tree.DecisionTreeClassifier()
model_Ex.fit(X_train, y_train)
print(model_Ex.score(X_test, y_test))
print(len(X_train))
print(len(X_test))
