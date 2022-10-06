# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 18:52:08 2022

@author: Rayan
"""

from flask import Flask, request,jsonify
import util
app=Flask(__name__)

@app.route("/get_location_names")

def get_location_names():
    response =jsonify({'location':util.get_location_names()})
    response.headers.add('Access-Control-Allow-Origin','*')
    
    return response
@app.route("/predict_home_price", methods=['POST'])
def predict_home_price():
    totl_sqft=float( request.form['total_sqft'])
    location=request.form['location']
    bhk=int(request.form['bhk'])
    bath=int(request.form['bath'])
    response=jsonify({
        'estimated_price':util.get_estimated_price(location, totl_sqft, bhk, bath)
            })
    response.headers.add('Access-Control-Allow-Origin',"*")
    return response
    
if __name__=="__mian__":
    util.load_saved_artfiacts()
    app.run()
