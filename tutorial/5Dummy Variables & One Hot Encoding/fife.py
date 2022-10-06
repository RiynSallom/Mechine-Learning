# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:00:08 2022

@author: ACER
"""
import pandas as pd
df = pd.read_csv("homeprices.csv")
print(df)
dummies = pd.get_dummies(df.town)
print(dummies)
merged = pd.concat([df,dummies],axis='columns')
print(merged)
final = merged.drop(['town','west windsor'], axis='columns')
print (final)
X = final.drop('price', axis='columns')
y = final.price
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(model.fit(X,y))
print(model.predict(X))
print(model.score(X,y))















#####################  EX   ####################
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# df=pd.read_csv('carprices.csv')
# print(df)
# dummies=pd.get_dummies(df['Car Model'])
# print(dummies)
# mergrd=pd.concat([df,dummies],axis='columns')
# print(mergrd)
# final=mergrd.drop(['Car Model'],axis='columns')
# print(final)
# final=final.drop(['Mercedez Benz C class'],axis='columns')
# print(final)
# x=final.drop(['Sell Price($)'],axis='columns')
# print(x)
# y=final['Sell Price($)']
# print(y)
# m=LinearRegression()
# m.fit(x,y)
# print(
# m.predict([[45000,4,0,0]]))
# print(m.predict([[86000,7,0,1]]))
# print(m.score(x,y))