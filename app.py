from os import environ
from flask import Flask, render_template, request
import requests
from epi import fox
from decisiontree import fox1
from svm import fox2

dataset=" "
app=Flask(__name__)
@app.route('/')
def index():
    data = {
        "title": 'Home Page',
        "msg":'Hello World from Flask for Python !!!',
        "me": environ.get('USERNAME')}
    return render_template('index.html', data = data)

@app.route('/dataset',methods=['GET', 'POST'])
def dataset():
    return render_template('dataset.html')


@app.route('/epilepsy',methods=['GET', 'POST'])
def epi():
    dataname=request.form.get('dataset')
    print("dataset",dataname)
    cm,sec=fox(dataname)
    #clf_load = joblib.load('saved_model.pkl')
    data = {
        "title": 'Epilepsy Page',
        "msg":'Hello World Using Neural Networks',
        "accuracy": cm,
        "time":sec}
    
    cm1,sec1=fox1(dataname)
    data1 = {
        "title": 'Epilepsy Page',
        "msg":'Hello World Using Decision Tree',
        "accuracy": cm1,
        "time":sec1}
    
    cm2,sec2=fox2(dataname)
    data2 = {
        "title": 'Epilepsy Page',
        "msg":'Hello World Using SVM',
        "accuracy": cm2,
        "time":sec2}

    return render_template('epi.html', data = data, data1= data1, data2= data2)


if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 3000
    app.run(HOST, PORT,debug=True)