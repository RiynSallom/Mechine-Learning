# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 19:28:06 2022

@author: Rayan
"""
import json
import pickle
import numpy as np
__data_columns=None
__location=None
__model=None
def get_estimated_price(loaction,sqft,bhk,bath):
    try:
        loc_index=__data_columns.index(loaction.lower())
    except:
        loc_index=-1
    x=np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1
    return round(__model.predict([x])[0],2)
def get_location_names():
    return __location
def load_saved_artfiacts():
    print("loading artfiacts ...start")
    global __data_columns
    global __location
    global __model
    with open("./artifacts/columns.json","r") as f:
        __data_columns=json.load(f)['data_col']
        __location=__data_columns[3:]
    with open("./artifacts/banglore_home_price_model.pickle","rb") as f:
        __model=pickle.load(f)
    print("loading artfiacts ..done")
if __name__=="__main__":
    load_saved_artfiacts()
    print(get_location_names())
    print(get_estimated_price('jp nagar', 1000, 2, 2))