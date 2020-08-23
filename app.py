from flask import Flask,render_template,url_for,request,jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import pickle
from pandas.io.json import json_normalize
import json

app = Flask(__name__)


model = joblib.load("model.pkl") # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

	Age=int(request.form['Age'])
	Academic_Year=int(request.form['Academic_Year'])
	CurrentGPA=float(request.form['CurrentGPA'])
	NoOfTimes=int(request.form['NoOfTimes'])
	Zscore=float(request.form['Zscore'])
	Gender=request.form['Gender']
	Hobby=request.form['Hobby']
	
	inputX=dict([('Age',Age),('Academic_Year',Academic_Year),('CurrentGPA',CurrentGPA),('NoOfTimes',NoOfTimes),('Zscore',Zscore),('Gender',Gender),('Hobby',Hobby)])

	

	print(inputX)
	
	
	json_=json.dumps(inputX)
	print(json_)

	query = pd.get_dummies(pd.DataFrame(eval(json_),index=[0]))
	
	query = query.reindex(columns=model_columns, fill_value=0)

	prediction = list(model.predict(query))
	print(prediction)

	if prediction==[0]:
		return render_template('cluster0.html')

	elif prediction==[1]:
		return render_template('cluster1.html')

	elif prediction==[2]:
		return render_template('cluster2.html')

	elif prediction==[3]:
		return render_template('cluster3.html')

	elif prediction==[4]:
		return render_template('cluster4.html')

	elif prediction==[5]:
		return render_template('cluster5.html')

	else:
		return render_template('cluster6.html')



	#return jsonify({'prediction': str(prediction)})

	


if __name__ == '__main__':
    app.run(debug=True)