"""
Fake News Detector Module
Core ML and NLP functionality

Author: Manas Baiswar
"""

import os
import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings('ignore')


class FakeNewsDetector:
    """Advanced Fake News Detection System using Machine Learning"""
    
    def __init__(self, model_path='models/model.pkl', vectorizer_path='models/vectorizer.pkl'):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.stop_words = set(stopwords.words('english'))
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
    
    def clean_text(self, text):
        """Advanced text preprocessing with NLP techniques"""
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def extract_features(self, text):
        """Extract additional linguistic features from text"""
        features = {}
        
        # Length-based features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # Sensational word indicators
        sensational_words = ['shocking', 'unbelievable', 'amazing', 'secret', 
                           'miracle', 'exposed', 'breaking', 'urgent', 'alert']
        features['sensational_count'] = sum(1 for word in sensational_words if word in text.lower())
        
        # Emotional language
        emotional_words = ['hate', 'love', 'fear', 'angry', 'happy', 'sad']
        features['emotional_count'] = sum(1 for word in emotional_words if word in text.lower())
        
        return features
    
    def train_model(self, fake_csv='data/Fake.csv', true_csv='data/True.csv'):
        """Train the fake news detection model"""
        print("Loading datasets...")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Load datasets or create sample data
        if not os.path.exists(fake_csv) or not os.path.exists(true_csv):
            print("Warning: Training data not found. Creating sample data...")
            self.create_sample_data()
        
        df_fake = pd.read_csv(fake_csv)
        df_true = pd.read_csv(true_csv)
        
        # Add labels
        df_fake['label'] = 0  # Fake
        df_true['label'] = 1  # Real
        
        # Combine datasets
        df = pd.concat([df_fake, df_true], ignore_index=True)
        
        # Keep only text and label columns
        if 'text' in df.columns:
            df = df[['text', 'label']]
        elif 'title' in df.columns:
            df['text'] = df['title'].fillna('') + ' ' + df.get('text', '').fillna('')
            df = df[['text', 'label']]
        
        # Remove duplicates and NaN values
        df = df.drop_duplicates(subset=['text'])
        df = df.dropna(subset=['text'])
        
        print(f"Total samples: {len(df)}")
        print(f"Fake news: {sum(df['label'] == 0)}, Real news: {sum(df['label'] == 1)}")
        
        # Clean text
        print("Preprocessing text...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'],
            test_size=0.2, random_state=42, stratify=df['label']
        )
        
        # TF-IDF Vectorization
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return accuracy
    
    def create_sample_data(self):
        """Create sample training data if files don't exist"""
        fake_samples = [
            "SHOCKING: Scientists discover miracle cure that doctors don't want you to know!",
            "BREAKING: Aliens confirmed by government officials in secret meeting!",
            "You won't believe what this celebrity said about politics!",
            "This one weird trick will make you rich overnight!",
            "Government hiding the truth about this dangerous conspiracy!",
            "EXPOSED: The secret they don't want you to see!",
            "URGENT: Share this before it gets deleted!",
            "AMAZING discovery that will change everything!",
            "Click here for the truth they're hiding from you!",
            "BREAKING NEWS: Unbelievable revelation about famous person!"
        ] * 100
        
        real_samples = [
            "According to recent studies published in Nature, researchers have made progress in understanding climate change.",
            "The university announced new findings in medical research conducted over three years.",
            "Economic indicators suggest steady growth in the manufacturing sector, experts report.",
            "Scientists at MIT have developed a new approach to renewable energy storage.",
            "Government officials announced policy changes following extensive public consultation.",
            "Research published in peer-reviewed journals indicates progress in cancer treatment.",
            "The Federal Reserve reported quarterly economic data showing moderate growth.",
            "Academic institutions collaborate on international research project.",
            "Public health officials provide updates on vaccination programs.",
            "Technology companies announce new developments in software security."
        ] * 100
        
        pd.DataFrame({'text': fake_samples}).to_csv('data/Fake.csv', index=False)
        pd.DataFrame({'text': real_samples}).to_csv('data/True.csv', index=False)
        print("Sample data created successfully!")
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        # Clean text
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return {'error': 'Text is empty after preprocessing'}
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([cleaned])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        # Get confidence
        confidence = probability[prediction] * 100
        
        # Extract features for analysis
        features = self.extract_features(text)
        
        result = {
            'prediction': 'REAL' if prediction == 1 else 'FAKE',
            'confidence': round(confidence, 2),
            'label': int(prediction),
            'probabilities': {
                'fake': round(probability[0] * 100, 2),
                'real': round(probability[1] * 100, 2)
            },
            'features': {
                'word_count': features['word_count'],
                'sensational_count': features['sensational_count'],
                'emotional_count': features['emotional_count']
            }
        }
        
        return result
    
    def scrape_article(self, url):
        """Scrape article text from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
            # Also try article tag
            if not text or len(text) < 100:
                article = soup.find('article')
                if article:
                    text = article.get_text()
            
            return text.strip() if text.strip() else "Could not extract text from URL"
        
        except requests.exceptions.RequestException as e:
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            return f"Error scraping URL: {str(e)}"
    
    def save_model(self):
        """Save trained model and vectorizer"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model saved to {self.model_path}")
        print(f"Vectorizer saved to {self.vectorizer_path}")
    
    def load_model(self):
        """Load trained model and vectorizer"""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.is_trained = True
            print("Model loaded successfully!")
            return True
        return False
