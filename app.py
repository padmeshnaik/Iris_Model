import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,url_for
import pickle



app=Flask(__name__)
model= pickle.load(open("model.pkl","rb"))


@app.route('/',methods=['POST','GET'])
def home():
    return render_template('Home.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    for x in prediction:
        ans=x
    
    if ans==0:
        label="Iris-setosa"
    elif ans==1:
        label="Iris-versicolor"
    else:
        label="Iris-virginica"


    return render_template('predict.html', prediction="The label should be {}".format(label))

@app.route('/demo',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Another':
            return redirect(url_for("home"))
        elif request.form['submit_button'] == 'Demo':
            return render_template("demo.html")
        else:
            return render_template("Home.html",prediction="Unknown site")