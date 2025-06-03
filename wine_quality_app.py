from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
with open('Regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_model.pkl', 'rb') as f:
    scaler = pickle.load(f)

type_map = {'white': 0, 'red': 1}

# Skewed columns to correct
skewed_features = ['type', 'fixed acidity', 'volatile acidity', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'sulphates']
# select numeric columns, excluding 'quality'
numeric_columns=['type', 'fixed acidity', 'volatile acidity', 'citric acid',
       'chlorides','total sulfur dioxide', 'density']


@app.route('/')
def home():
    return render_template('form.html', prediction_text='')  # Assuming index.html is used

@app.route('/predict', methods=['POST'])
def index():
    input_df = {
        'type': request.form['type'],
        'fixed acidity': float(request.form['fixed acidity']),
        'volatile acidity': float(request.form['volatile acidity']),
        'citric acid': float(request.form['citric acid']),
        'residual sugar': float(request.form['residual sugar']),  
        'chlorides': float(request.form['chlorides']),
        'free sulfur dioxide': float(request.form['free sulfur dioxide']),
        'total sulfur dioxide': float(request.form['total sulfur dioxide']),
        'density': float(request.form['density']),
        'pH': float(request.form['pH']),
        'sulphates': float(request.form['sulphates']),
        'alcohol': float(request.form['alcohol'])
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_df]) 

    # Map categorical feature
    input_df['type'] = input_df['type'].map(type_map)

    # Scale numeric columns
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    # Apply log1p transformation to correct skewness
    for col in skewed_features:
        input_df[col] = np.log1p(input_df[col])



    # Make prediction
    result = model.predict(input_df)

    # Render result
    return render_template('index.html', prediction_text=f'Predicted class: {result[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
