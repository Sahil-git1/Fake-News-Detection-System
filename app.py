import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import requests
import schedule
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import threading
import json
import os
MAX_WORDS = 10000
MAX_LEN = 200
MODEL_PATH = 'fake_news_model.h5'
TRAINING_DATA = 'data.csv'
NEWS_API_KEY = 'c73e3ff50e6640f59d36a16924411726' 
CACHE_FILE = 'news_cache.json'

# Flask app initialization
app = Flask(__name__)
model = None
tokenizer = None
is_updating = False
recent_news = []

def fetch_initial_news():
    global recent_news
    
    print("Fetching initial news data...")
    
    if os.path.exists(TRAINING_DATA):
        df_existing = pd.read_csv(TRAINING_DATA)
    else:
        df_existing = pd.DataFrame(columns=['text', 'label'])

    sources = ['bbc-news', 'reuters', 'the-wall-street-journal', 'the-washington-post']
    all_articles = []
    
    for source in sources:
        url = f'https://newsapi.org/v2/top-headlines?sources={source}&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        
        if response.status_code == 200:
            news_data = response.json()
            articles = news_data.get('articles', [])
            all_articles.extend(articles)
    
    new_articles = []
    for article in all_articles:
        if article['description']:
            new_articles.append({
                'text': article['description'],
                'label': 'real',  
                'title': article['title'],
                'source': article['source']['name'],
                'url': article['url'],
                'publishedAt': article['publishedAt']
            })

    recent_news = sorted(new_articles[:25], 
                        key=lambda x: x['publishedAt'], 
                        reverse=True)
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(recent_news, f)
    
    training_articles = [{
        'text': article['text'],
        'label': article['label']
    } for article in new_articles]
    
    df_new = pd.DataFrame(training_articles)
    df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    df_updated.to_csv(TRAINING_DATA, index=False)
    
    print(f"Dataset updated with {len(new_articles)} new articles")
    return df_updated

def load_and_preprocess_data(df):
    """Load and preprocess the dataset"""
    X = df['text'].values
    y = df['label'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y

def create_tokenizer(texts):
    """Create and fit tokenizer"""
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def build_model(vocab_size):
    """Create the LSTM model"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=MAX_LEN),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def train_model(df):
    """Train the fake news detection model"""
    global model, tokenizer, is_updating
    
    is_updating = True
    print("Training model...")
    
    X, y = load_and_preprocess_data(df)
    
    tokenizer = create_tokenizer(X)
    
    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=MAX_LEN)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, random_state=42
    )
    
    model = build_model(MAX_WORDS)
    
    checkpoint = ModelCheckpoint(
        MODEL_PATH, monitor='val_accuracy', 
        save_best_only=True, mode='max'
    )
    
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=32,
        callbacks=[checkpoint]
    )
    
    is_updating = False
    print("Model training completed")

def update_system():
    """Update the entire system"""
    df = fetch_initial_news()
    train_model(df)

def schedule_updates():
    """Schedule periodic updates"""
    schedule.every(24).hours.do(update_system)
    
    while True:
        schedule.run_pending()
        time.sleep(3600)

# Flask 
@app.route('/')
def home():
    return render_template('index.html', news_articles=recent_news)

@app.route('/predict', methods=['POST'])
def predict():
    if is_updating:
        return jsonify({'error': 'Model is currently updating. Please try again later.'})
    
    text = request.json['text']
    
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    
    prediction = model.predict(padded)[0][0]
    
    return jsonify({
        'prediction': 'fake' if prediction < 0.5 else 'real',
        'confidence': float(prediction)
    })


html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --background-color: #ecf0f1;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--primary-color);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .detector-section {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            animation: fadeIn 1s ease-in;
        }
        
        textarea {
            width: 100%;
            padding: 1rem;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-bottom: 1rem;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        textarea:focus {
            border-color: var(--secondary-color);
            outline: none;
        }
        
        button {
            background-color: var(--secondary-color);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1rem;
            transition: transform 0.2s ease, background-color 0.3s ease;
        }
        
        button:hover {
            transform: translateY(-2px);
            background-color: #2980b9;
        }
        
        #result {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 5px;
            animation: slideIn 0.5s ease-out;
        }
        
        .result-real {
            background-color: #a8e6cf;
            border-left: 4px solid #1abc9c;
        }
        
        .result-fake {
            background-color: #ffd3b6;
            border-left: 4px solid #e74c3c;
        }
        
        .updating {
            background-color: #fff3cd;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 5px;
            display: none;
            animation: pulse 2s infinite;
        }
        
        .news-section {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-in;
        }
        
        .news-item {
            padding: 1rem;
            border-bottom: 1px solid #ddd;
            transition: transform 0.2s ease;
        }
        
        .news-item:hover {
            transform: translateX(10px);
            background-color: #f8f9fa;
        }
        
        .news-title {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: bold;
        }
        
        .news-source {
            color: var(--secondary-color);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        .news-date {
            color: #666;
            font-size: 0.8rem;
        }
        
        @keyframes slideIn {
            from {
                transform: translateY(-20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="header animate__animated animate__fadeIn">
        <h1>Fake News Detector</h1>
        <p>Verify the authenticity of news articles using AI</p>
    </div>
    
    <div class="container">
        <div class="detector-section animate__animated animate__fadeInUp">
            <div id="updating-message" class="updating">
                <strong> Model Update in Progress</strong>
                <p>Please wait while we update our detection system...</p>
            </div>
            
            <h2>Check News Authenticity</h2>
            <textarea id="news-text" rows="5" 
                      placeholder="Enter news text here to verify its authenticity..."></textarea>
            <button onclick="checkNews()">Analyze News</button>
            <div id="result"></div>
        </div>
        
        <div class="news-section animate__animated animate__fadeInUp">
            <h2>Latest News Articles</h2>
            {% for article in news_articles %}
            <div class="news-item animate__animated animate__fadeIn">
                <a href="{{ article.url }}" target="_blank" class="news-title">
                    {{ article.title }}
                </a>
                <div class="news-source">
                    Source: {{ article.source }}
                </div>
                <div class="news-date">
                    Published: {{ article.publishedAt }}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <script>
        function checkNews() {
            const text = document.getElementById('news-text').value;
            const resultDiv = document.getElementById('result');
            
            // Show loading state
            resultDiv.innerHTML = '<div class="updating">Analyzing...</div>';
            resultDiv.style.display = 'block';
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text: text}),
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('updating-message').style.display = 'block';
                    resultDiv.style.display = 'none';
                } else {
                    document.getElementById('updating-message').style.display = 'none';
                    
                    const confidence = (data.confidence * 100).toFixed(2);
                    const resultClass = data.prediction === 'real' ? 
                        'result-real' : 'result-fake';
                    
                    resultDiv.className = `${resultClass} animate__animated animate__fadeIn`;
                    resultDiv.innerHTML = `
                        <strong>Prediction: ${data.prediction.toUpperCase()}</strong>
                        <br>
                        Confidence: ${confidence}%
                    `;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                resultDiv.innerHTML = 'An error occurred while analyzing the text.';
            });
        }
    </script>
</body>
</html>
"""

def create_template_directory():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    with open('templates/index.html', 'w') as f:
        f.write(html_template)

if __name__ == '__main__':
    print("Starting system setup...")
    
    create_template_directory()
    
    update_system()
    
    update_thread = threading.Thread(target=schedule_updates)
    update_thread.daemon = True
    update_thread.start()
    print("Starting web server...")
    app.run(debug=True)