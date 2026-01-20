"""
Configuration settings for the application

Author: Manas Baiswar
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    
    # API settings
    API_KEY = os.getenv('API_KEY', 'your-default-api-key-here')
    API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', 100))
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')
    VECTORIZER_PATH = os.getenv('VECTORIZER_PATH', 'models/vectorizer.pkl')
    MAX_FEATURES = int(os.getenv('MAX_FEATURES', 5000))
    NGRAM_RANGE = tuple(map(int, os.getenv('NGRAM_RANGE', '1,3').split(',')))
    
    # Data settings
    FAKE_CSV = os.getenv('FAKE_CSV', 'data/Fake.csv')
    TRUE_CSV = os.getenv('TRUE_CSV', 'data/True.csv')
    
    # External API keys (optional)
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
