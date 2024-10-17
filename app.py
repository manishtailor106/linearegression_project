import numpy as np
from collections import Mapping
from collections.abc import Mapping,Sequence
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)
model = pickle.load(open('/content/drive/My Drive/model.pkl','rb'))
run_with_ngrok(app)

@app.route('/')
def home():

    return render_template("index.html")

@app.route('/predict',methods=['GET'])
def predict():


    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))

    prediction = model.predict([[exp]])


    return render_template('index.html', prediction_text='Regression Model  has predicted salary for given experinace is : {}'.format(prediction))

if __name__ == '__main__':
    app.run()
