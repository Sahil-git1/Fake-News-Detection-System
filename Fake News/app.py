from flask import Flask, render_template, request, jsonify
import joblib
from model_training import ensure_dataset_exists, ensure_model_exists
import os

app = Flask(__name__, 
            static_folder=os.path.abspath('static'),
            template_folder=os.path.abspath('templates'))
# Ensure the dataset and model exist before loading
ensure_dataset_exists()
ensure_model_exists()

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_news', methods=['POST'])
def check_news():
    data = request.json
    news_text = data['text']
    
    # Vectorize the input text
    vectorized_text = vectorizer.transform([news_text])
    
    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    
    return jsonify({'result': 'fake' if prediction == '0' else 'real'})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=3000, debug=True)