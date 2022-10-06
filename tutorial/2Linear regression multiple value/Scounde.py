# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 04:37:15 2022

@author: ACER
"""

import pandas as pd
from sklearn import linear_model
import math as m
import matplotlib.pyplot as plt
df=pd.read_csv('homeprices.csv')
print(df)
print(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(int(df.bedrooms.median()))
print(df)
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
print(reg)
print(reg.coef_)
print(reg.intercept_)
reg.predict([[3000,3,40]])
print(137.25*3000+-26025*3+-6825*40+383724.9999999998)#m*x+m*x+m*x+b
print(reg.predict([[2500,4,5]]))
print(137.25*2500+-26025*4+-6825*5+383724.9999999998)
#//////////////////////////////////////////////////////
from word2number import w2n
d=pd.read_csv("hiring.csv")
print(d)
d.experience=d.experience.fillna("zero")
print(d)
d.experience=d.experience.apply(w2n.word_to_num)
print(d)
s=m.floor(d["test_score(out of 10)"].mean())
d["test_score(out of 10)"]=d["test_score(out of 10)"].fillna(s)
print(d)
regex=linear_model.LinearRegression()
regex.fit(d[["experience","test_score(out of 10)","interview_score(out of 10)"]],d["salary($)"])
print(regex.predict([[2,9,6]]))
print(regex.predict([[12,10,10]]))

