import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import requests
from datetime import datetime, timedelta

# Check if dataset exists, if not create a simple one
def ensure_dataset_exists():
    if not os.path.exists('fake_news_dataset.csv'):
        print("Dataset not found. Creating a simple dataset...")
        df = pd.DataFrame({
            'text': [
                'This is a fake news article about aliens',
                'Scientists discover new planet in solar system',
                'Conspiracy theory claims Earth is flat',
                'New study shows benefits of exercise'
            ],
            'label': ['fake', 'real', 'fake', 'real']
        })
        df.to_csv('fake_news_dataset.csv', index=False)
        print("Simple dataset created.")
    else:
        print("Dataset found.")

# Load dataset
def load_dataset():
    return pd.read_csv('fake_news_dataset.csv')

def train_model():
    df = load_dataset()
    
    # Preprocess the data
    X = df['text']
    y = df['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    # Save the model and vectorizer
    joblib.dump(model, 'fake_news_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

def update_dataset():
    # Use a news API to fetch recent articles
    api_key = "Your Api Key"
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
    response = requests.get(url)
    articles = response.json()['articles']

    # Process and add new articles to the dataset
    new_data = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Combine title and description, handling cases where either might be None
        text = ' '.join(filter(None, [title, description]))
        
        if text:  # Only add the article if there's some text content
            new_data.append({
                'text': text,
                'label': 1  # Assume all API articles are real for simplicity
            })

    df = load_dataset()
    df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
    df.to_csv('fake_news_dataset.csv', index=False)
    print(f"Dataset updated with {len(new_data)} new articles.")

def ensure_model_exists():
    if not os.path.exists('fake_news_model.joblib') or not os.path.exists('vectorizer.joblib'):
        print("Model or vectorizer not found. Training new model...")
        train_model()
    else:
        print("Model and vectorizer found.")

def update_and_retrain():
    update_dataset()
    train_model()
    print("Model updated and retrained.")

if __name__ == "__main__":
    ensure_dataset_exists()
    ensure_model_exists()
    # Uncomment the following line to update the dataset and retrain the model
    update_and_retrain()
