# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 02:15:24 2022

@author: Rayan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score









df=pd.read_csv("eda_data.csv")
print(df.head())

#choose relevant columns
df.columns
df_model=df[['avg_salary','Rating','Size','Type of ownership','Industry', 'Sector', 'Revenue','num_comp',
           'hourly', 'employer_provided','job_state', 'same_state', 'age', 'python_yn',
           'spark', 'aws', 'excel','job_simp', 'seniority', 'desc_len']]
#get dummy data

df_dum=pd.get_dummies(df_model)

#traing test split
x=df_dum.drop('avg_salary',axis=1)
y=df_dum.avg_salary
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=10)
#multiple linear regression
s_sm = x= sm.add_constant(x)
model= sm.OLS(y,s_sm)
model.fit().summary()

lm=LinearRegression()
lm.fit(X_train,y_train)

cross_val_score(lm,X_train,y_train,scoring='neg_mean_squared_error')

#lasso regression

#rendom forest


#trune models GrideaechCV

#test ensmble