# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:36:04 2022

@author: ACER
"""
import math
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.linear_model import LinearRegression

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def prediction_function(age):
        z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
        y = sigmoid(z)
        return y
df_=pd.read_csv("insurance_data.csv")
print(df_.head())
sns.regplot(x=df_.age, y=df_.bought_insurance, data=df_, logistic=True, ci=None,scatter_kws={'color': 'black'}, line_kws={'color': 'red'})
X_train, X_test, y_train, y_test = train_test_split(df_[['age']],df_.bought_insurance,train_size=0.8)
model = LogisticRegression()
model.fit(X_train,y_train)
age = 35
print(prediction_function(age))
age = 43
print(prediction_function(age))
################################################################
df = pd.read_csv("HR_comma_sep.csv")
print(df.head())
left = df[df.left==1]
print(left.shape)
retained = df[df.left==0]
print(retained.shape)
print(df.groupby('left').mean())
pd.crosstab(df.salary,df.left).plot(kind='bar')
plt.show()
pd.crosstab(df.Department, df.left).plot(kind='bar')
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
print(subdf.head())
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
print(df_with_dummies.head())
df_with_dummies.drop('salary',axis='columns',inplace=True)
print(df_with_dummies.head())
X = df_with_dummies
print(X.head())
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))

plt.show()
df_y=pd.read_csv("carprices.csv")
print(df_y.columns)
plt.scatter(df_y.Mileage, df_y['Sell Price($)'])
plt.show()
plt.scatter(df_y['Age(yrs)'], df_y['Sell Price($)'])
plt.show()
x=df_y[['Mileage','Age(yrs)']]
y=df_y[['Sell Price($)']]
x_t,x_s,y_t,y_s=train_test_split(x,y,test_size=0.2)
print(len(x))
print(len(x_t))
c=LinearRegression()
c.fit(x_t,y_t)
print(c.predict(x_s))
print(c.score(x_s, y_s))
