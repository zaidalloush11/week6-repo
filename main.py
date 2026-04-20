import pandas as pd
from preprocessor import TextCleaner
from model_trainer import SentimentModel
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
print("Loading Data...")
df = pd.read_csv("week6-repo/archive/sentiment.csv", encoding='latin1', low_memory=False)

# 2. Prepare Columns and Balance Data
df = df.rename(columns={'Review': 'text', 'Sentiment': 'sentiment'})
df = df.dropna(subset=['text', 'sentiment'])
df['sentiment'] = df['sentiment'].astype(str).str.capitalize()

print("Checking Data Balance...")
print(df['sentiment'].value_counts())

min_size = df['sentiment'].value_counts().min()
df = df.groupby('sentiment').sample(n=min_size, random_state=42)

print(f"New Balanced Size: {min_size} rows per category.")

# 3. Clean Data using our module
print("Cleaning text data (this might take a few seconds)...")
cleaner = TextCleaner()
df['cleaned_text'] = df['text'].apply(cleaner.clean)

# 4. Train Model using our module
print("Training Model...")
trainer = SentimentModel()
X_test, y_test = trainer.prepare_and_train(df)

# 5. Evaluate
y_pred = trainer.predict(X_test)
print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# 6. Live Testing Loop
print("\n" + "="*30)
print("LIVE SENTIMENT TESTER")
print("Type 'quit' to stop")
print("="*30)

while True:
    user_input = input("\nEnter a review to test: ")
    if user_input.lower() == 'quit':
        break
    
    cleaned_input = cleaner.clean(user_input)
    prediction = trainer.predict([cleaned_input])
    
    print(f"AI Analysis: This sentiment is {prediction[0]}")