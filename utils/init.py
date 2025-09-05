# utils/__init__.py
import re
import pandas as pd
from bs4 import BeautifulSoup
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_html(text):
    """Clean HTML tags using BeautifulSoup"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def expand_contractions(text):
    """Expand contractions like don't to do not"""
    if pd.isna(text) or text is None or text == "":
        return ""
    try:
        return contractions.fix(text)
    except:
        return text

def preprocess_text(text):
    """Comprehensive text preprocessing"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    if text.strip() == "":
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

def remove_stopwords(text, custom_stopwords=None):
    """Remove stopwords with custom additions"""
    if pd.isna(text) or text is None or text == "":
        return ""
    
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """Apply lemmatization with POS tagging"""
    if pd.isna(text) or text is None or text == "":
        return ""
    
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def full_clean_pipeline(text, custom_stopwords=None):
    """Complete text cleaning pipeline with robust NaN handling"""
    if pd.isna(text) or text is None:
        return "no content"
    
    text = str(text)
    if text.strip() == "":
        return "no content"
    
    text = clean_html(text)
    text = expand_contractions(text)
    text = preprocess_text(text)
    text = remove_stopwords(text, custom_stopwords)
    text = lemmatize_text(text)
    
    if text.strip() == "":
        return "no content"
    
    return text

def evaluate_model(model, X_test, y_test, model_name=""):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    
    print(f"=== {model_name} Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def save_model(model, filename):
    """Save model to file"""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load model from file"""
    return joblib.load(filename)