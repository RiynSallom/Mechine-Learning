# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 04:46:45 2022

@author: Rayan
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor 
import pickle
import json
import xgboost 
df=pd.read_csv("Book1.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.groupby('area_type')['area_type'].agg('count'))
#remove unnecessary columns 
df2=df.drop(['area_type','society','balcony','availability'],axis=1)
print(df2.head())
print(df2.isnull().sum())
#remove rows containing null values
df3=df2.dropna()
print(df3.isnull().sum())
print(df3['size'].unique())
#add bhk column(the number of rooms) 
df3['bhk']=df3['size'].apply(lambda x:int(x.split(' ')[0]))
print(df3.head())
print(df2.loc[30])
print(df3['bhk'].unique())
print(df3[df3['bhk']>20])
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
#Parsed float data out of total_sqft and get avg (total_sqft    2100 - 2850)
df3[~df3['total_sqft'].apply(is_float)]
df4=df3.copy()
def convert_to_avg(x):
    splt=x.split(' - ')
    if len(splt)==2:
        return (float(splt[0])+float(splt[-1]))/2
    try:
        return float(x)
    except:
        return None
print(convert_to_avg('2100 - 2850'))
df4['total_sqft']=df4['total_sqft'].apply(convert_to_avg)
print(df4.loc[30])
df5=df4.copy()
#add column price_per_sqft contain price per sqft
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
print(df5.head())
print(len(df5.location.unique()))
df5.location=df5.location.apply(lambda x:x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)
location_stats_less_than_10=location_stats[location_stats<=10]
print(location_stats_less_than_10)
#update column location to count of home less than 10 to "other"
df5.location=df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))
print('iiiiiiiiiiiiiiiiiiiiiiiiiii')
print(df5.describe())
print(df5[df5.total_sqft>4000])
print(df5.shape)
plt.scatter(df5.total_sqft, df5.bhk)
plt.xlabel('price_per_sqft')
plt.ylabel("bhk")
plt.show()
#remove outlier from dataset
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        print(f'key={key}, sub= {subdf}'.format(key, subdf))
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df6=df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.head())
print(df6.describe())
print(df6.shape)
df7=remove_pps_outliers(df6)
plt.scatter(df7.price_per_sqft, df7.bhk)
plt.xlabel('price_per_sqft')
plt.ylabel("bhk")
plt.show()
print(df7.head())
print(df7.describe())
print(df7.shape)
def plot_scatter_char(df,location):
    bhk2=df[(df.location==location)&(df.bhk==2)]
    bhk3=df[(df.location==location)&(df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price_per_sqft,color='blue',label='2 bhk',s=100)
    plt.scatter(bhk3.total_sqft,bhk3.price_per_sqft,marker='*',color='red',label='3 bhk',s=100)
    plt.xlabel('total squre feet area')
    plt.ylabel('price per square feer')
    plt.title(location)
    plt.legend()
    plt.show()
plot_scatter_char(df7,'Hebbal')
print(df.location.unique())
#remove remove bhk outliers 
def remove_bhk_outliers(df):
    excluede_indies=np.array([])
    for loc,loc_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in loc_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
                }
        for bhk,bhk_df in loc_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
              excluede_indies=np.append(excluede_indies,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(excluede_indies,axis='index')
print(df7.head())
df8=remove_bhk_outliers(df7)
print(df8.shape)
plot_scatter_char(df8,'Hebbal')
matplotlib.rcParams['figure.figsize']=(20,20)
plt.hist(df8.price_per_sqft,rwidth=0.8) 
plt.show() 
#remove row containing number of bath > bhk+2
df9=df8[df8.bath<df8.bhk+2] 
print(df9.shape)
plt.hist(df9.price_per_sqft,rwidth=0.8)
df10=df9.drop(['size','price_per_sqft'],axis='columns')
df10.location.unique()
df11=pd.concat([df10,pd.get_dummies(df10.location).drop('other',axis='columns')],axis='columns')
df11.location.unique()
print(df11.head())
df12=df11.drop('location',axis='columns')
print(df12.shape)
x=df12.drop('price',axis='columns')
y=df12.price



X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)
lr_crf=LinearRegression()
lr_crf.fit(X_train,y_train)
print(lr_crf.score(X_test, y_test))
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
print(cross_val_score(LinearRegression(),x,y,cv=cv))

def find_best_model_using_gridsearchcv(x,y):
    algos={
            'LinearRegression':{
                'model':LinearRegression(),
                'params':{
                    'normalize':[True,False]
                    }},
            'lasson':{
                'model':Lasso(),
                'params':{
                    'alpha':[1,2],
                    'selection':['random','cyclic']
                    } },
            'decision_tree':{
                'model':DecisionTreeRegressor(),
                'params':{
                    'criterion':['mse','frieedman_mse'],
                    'splitter':['best','random']
                    }
                }  }
    scores=[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs=GridSearchCV(config['model'], config['params'],cv=cv,return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
            })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(find_best_model_using_gridsearchcv(x, y))
def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(x.columns==location)[0][0]
    X=np.zeros(len(x.columns))
    X[0]=sqft
    X[1]=bath
    X[2]=bhk
    if loc_index>=0:
        X[loc_index]=1
    return lr_crf.predict([X])[0]
print(predict_price('Indira Nagar', 1000, 3, 3))
print(df12.columns)
print(predict_price('1st Phase JP Nagar',1000,2,3))

with open('banglore_home_price_modelttttttttttttttt.pickle','wb') as f:
    pickle.dump(lr_crf, f)
col={
     'data_col' : [coll.lower() for coll in x.columns]
     }
with open('columnsttttttttttttttt.json','w') as f:
    f.write(json.dumps(col))
