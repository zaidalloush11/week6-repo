from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class SentimentModel:
    def __init__(self):
        # 1. Increase max_features to 15,000 so it can learn more words.
        # 2. Lower min_df to 3 so it catches slightly less common powerful words.
        self.tfidf = TfidfVectorizer(
            max_features=15000, 
            ngram_range=(1, 2), 
            min_df=3,
            max_df=0.8
        )
        
        # 3. Increase C to 2.0 (Take the brakes off!).
        # 4. Remove class_weight='balanced' because we already balance the data in main.py.
        self.model = LogisticRegression(C=2.0, max_iter=1000)

    def prepare_and_train(self, df):
        X = df['cleaned_text']
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
        
        return X_test, y_test

    def predict(self, text_series):
        X_tfidf = self.tfidf.transform(text_series)
        return self.model.predict(X_tfidf)