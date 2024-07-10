import os
import schedule
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from datetime import datetime
import requests

# Replace 'YOUR_API_KEY' with your actual News API key
NEWS_API_KEY = "Your Api Key"
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'

def fetch_news_articles( language='en', category='general', page_size=100):
    params = {
        'apiKey': NEWS_API_KEY,
        'category': category,
        # 'country': country,
        'pageSize': page_size
    }
    
    response = requests.get(NEWS_API_URL, params=params)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        
        # Extract title and content from each article
        processed_articles = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', '')
            if title and content:
                processed_articles.append({
                    'text': f"{title} {content}",
                    'label': 0  # Assuming all fetched articles are real news (label 0)
                })
        
        return pd.DataFrame(processed_articles)
    else:
        print(f"Error fetching news: {response.status_code}")
        return pd.DataFrame()

def fetch_new_data():
    # Fetch real news
    real_news = fetch_news_articles()
    
    # For fake news, you might want to use a different source or generate synthetic data
    # For this example, we'll just create some dummy fake news
    fake_news = pd.DataFrame([
        {"text": "Aliens confirmed to be living among us, government admits", "label": 1},
        {"text": "New study shows chocolate is the key to immortality", "label": 1},
        {"text": "Scientists discover that the Earth is actually flat", "label": 1},
    ])
    
    # Combine real and fake news
    new_data = pd.concat([real_news, fake_news], ignore_index=True)
    
    return new_data

def update_dataset_and_model():
    print(f"Updating dataset and model at {datetime.now()}")
    
    # Load the existing dataset
    df = pd.read_csv(os.path.join('data', 'fake_news_dataset.csv'))
    
    # Fetch new data
    new_data = fetch_new_data()
    
    # Append new data to the existing dataset
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Save the updated dataset
    df.to_csv(os.path.join('data', 'fake_news_dataset.csv'), index=False)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train the model
    clf = MultinomialNB()
    clf.fit(X_train_vectorized, y_train)
    
    # Save the model and vectorizer
    joblib.dump(clf, os.path.join('models', 'fake_news_model.joblib'))
    joblib.dump(vectorizer, os.path.join('models', 'vectorizer.joblib'))
    
    print("Dataset and model updated successfully")

# Schedule the update task to run every 24 hours
schedule.every(1).hours.do(update_dataset_and_model)

# Run the scheduled tasks
while True:
    schedule.run_pending()
    time.sleep(1)