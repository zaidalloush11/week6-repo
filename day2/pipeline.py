import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==========================================
# 1. Data Loading & Exploration
# ==========================================
def load_dataset(filepath):
    """Load dataset from a CSV file."""
    print(f"Loading dataset from {filepath}...")
    # Using latin1 and low_memory=False based on our Day 1 findings!
    df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
    return df

def explore_dataset(df):
    """Inspect missing values, sample rows, and class distribution."""
    print("\n--- Dataset Exploration ---")
    print(f"Total Rows: {len(df)}")
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nClass Distribution:")
    if 'Sentiment' in df.columns:
        print(df['Sentiment'].value_counts())
    elif 'sentiment' in df.columns:
        print(df['sentiment'].value_counts())
        
    print("\nSample Rows:")
    print(df.head(2))
    print("---------------------------\n")

# ==========================================
# 2. Text Preprocessing
# ==========================================
def clean_text(text):
    """
    Clean text: lowercasing, removing special characters, and extra spaces.
    (This is the exact function you will need for your live testing).
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower() # Lowercase
    text = re.sub(r'<.*?>', ' ', text) # Remove HTML tags if any
    text = re.sub(r'[^a-z\s]', '', text) # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    
    # Note: For production, you can add NLTK stopword removal here
    return text

def preprocess_dataset(df, text_column='text'):
    """Apply clean_text to the entire dataset."""
    print("Cleaning dataset text...")
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # Handle column names dynamically based on our specific dataset
    if 'Review' in df_clean.columns:
        df_clean = df_clean.rename(columns={'Review': 'text'})
    if 'Sentiment' in df_clean.columns:
        df_clean = df_clean.rename(columns={'Sentiment': 'sentiment'})
        
    df_clean = df_clean.dropna(subset=['text', 'sentiment'])
    df_clean['cleaned_text'] = df_clean['text'].apply(clean_text)
    df_clean['sentiment'] = df_clean['sentiment'].astype(str).str.capitalize()
    return df_clean

# ==========================================
# 3. Feature Extraction & Splitting
# ==========================================
def extract_features(df, text_column='cleaned_text', method='tfidf'):
    """
    Convert text to numerical features using TF-IDF or BoW.
    (Word Embeddings like Word2Vec can be added here, but TF-IDF/BoW 
    returns the specific vectorizer object needed for vectorizer.pkl)
    """
    print(f"Extracting features using {method.upper()}...")
    
    if method.lower() == 'tfidf':
        # Advanced TF-IDF with n-grams
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5)
    elif method.lower() == 'bow':
        vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2))
    else:
        raise ValueError("Method must be 'tfidf' or 'bow'")
        
    X = vectorizer.fit_transform(df[text_column])
    return X, vectorizer

def split_data(X, y):
    """Split into 80/20 train and test sets."""
    print("Splitting data into 80% Train, 20% Test...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. Model Training & Evaluation
# ==========================================
def train_model(X_train, y_train, model_name='Logistic Regression'):
    """Train Naive Bayes or Logistic Regression."""
    print(f"Training {model_name} Model...")
    
    if model_name == 'Logistic Regression':
        # Advanced Hyperparameter tuning: C=2.0
        model = LogisticRegression(C=2.0, max_iter=1000)
    elif model_name == 'Naive Bayes':
        model = MultinomialNB()
    else:
        raise ValueError("model_name must be 'Logistic Regression' or 'Naive Bayes'")
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Calculate metrics and draw confusion matrix."""
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    # Use average='weighted' for multi-class (Positive, Negative, Neutral)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n--- Model Performance ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Draw Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    labels = sorted(list(set(y_test)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Sentiment Analysis Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('day2_confusion_matrix.png')
    print("Confusion matrix saved as 'day2_confusion_matrix.png'")
    plt.close()

# ==========================================
# 5. Deployment (Save, Load, Predict)
# ==========================================
def save_model(model, vectorizer):
    """Save the trained model and vectorizer using pickle."""
    print("\nSaving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Saved successfully as 'model.pkl' and 'vectorizer.pkl'!")

def load_model():
    """Load the saved model and vectorizer."""
    print("Loading saved model and vectorizer...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict(text, model, vectorizer):
    """Clean, transform, and predict sentiment for new text."""
    # 1. Clean the text using the exact same function
    cleaned = clean_text(text)
    
    # 2. Transform using the loaded vectorizer
    vectorized_text = vectorizer.transform([cleaned])
    
    # 3. Predict
    prediction = model.predict(vectorized_text)
    return prediction[0]

# ==========================================
# 6. Main Execution Block
# ==========================================
# --- Main Execution to test everything ---
if __name__ == "__main__":
    filepath = "/home/cybersecurity/training/week6/week6-repo/archive/sentiment.csv" 
    
    print("Running Pipeline...")
    df_raw = load_dataset(filepath)
    
    # Preprocess all data without throwing any away
    df_clean = preprocess_dataset(df_raw)
    
    # Extract Features
    X, vectorizer = extract_features(df_clean, method='tfidf')
    y = df_clean['sentiment']
    
    # Split, Train, and Evaluate
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train using the new balanced weights
    model = train_model(X_train, y_train, model_name='Logistic Regression')
    evaluate_model(model, X_test, y_test)
    
    # Save Deployment Files
    save_model(model, vectorizer)
    
    # Final Test
    print("\n" + "="*40)
    print("TESTING DEPLOYMENT WORKFLOW")
    print("="*40)
    loaded_model, loaded_vectorizer = load_model()
    
    test_sentences = [
        "it is amazing",                  # Should be Positive
        "i received the package today",   # Should be Neutral
        "this is a total disaster"        # Should be Negative
    ]
    
    for sentence in test_sentences:
        result = predict(sentence, loaded_model, loaded_vectorizer)
        print(f"Text: '{sentence}'")
        print(f"Predicted Sentiment: {result}\n")
