# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:14:13 2022

@author: ACER
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
#from sklearn.externals import joblib
df=pd.read_csv("homeprice.csv")
print(df)
print(df.columns)
plt.scatter(df.area, df.price)
reg=linear_model.LinearRegression(copy_X=True,fit_intercept=True, n_jobs=1,normalize=False)
print(reg.fit(df[['area']],df.price))
print(reg.predict([[3300]]))
print(reg.coef_)
print(reg.intercept_)
print(reg.coef_*3300+reg.intercept_)
plt.xlabel('area')
plt.ylabel('price')
plt.plot(df.area, reg.predict(df[['area']]))
plt.show()
d=pd.read_csv('areas.csv')
print(d.head())
p=reg.predict(d)
d['prices']=p
print(d)
d.to_csv('products.csv',index=False)
############################/////////////////////////////////////////////////
d=pd.read_csv("canada_per_capita_income.csv")
print(d.head())
plt.scatter(d.year, d['per capita income (US$)'])
re=linear_model.LinearRegression()
re.fit(d[['year']],d['per capita income (US$)'])
re.predict([[2020]])
plt.xlabel('yare')
plt.ylabel('price')
plt.plot(d.year, re.predict(d[['year']]))
print(re.predict([[2020]]))

with open('model_pickle','wb')as f:
    pickle.dump(reg, f)
with open('model_pickle','rb')as f:
    m=pickle.load(f)
print(m.predict([[5000]]))
