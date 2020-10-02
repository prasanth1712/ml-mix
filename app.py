# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 15:20:14 2020

@author: pr
"""
import numpy as np
from flask import Flask , request , jsonify , render_template
import pickle

app = Flask(__name__)
adm  = pickle.load(open('admission_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():   
    int_features = [int(x) for x in request.form.values()]
    int_features = [np.array(int_features)]
    out = adm.predict(int_features)

    return render_template('index.html',pr=out)

if __name__ == "__main__":
    app.run(debug=True)
    

