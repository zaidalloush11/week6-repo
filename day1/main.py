import pandas as pd
from preprocessor import TextCleaner
from model_trainer import SentimentModel
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. Load Data
print("Loading Data...")
df = pd.read_csv("/home/cybersecurity/training/week6/week6-repo/archive/sentiment.csv", encoding='latin1', low_memory=False)

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

# ==========================================
# VISUALIZATION CODE
# ==========================================
print("Generating visualizations...")
labels = ["Negative", "Neutral", "Positive"]

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Sentiment Analysis Confusion Matrix')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.savefig('confusion_matrix.png') 
plt.close()

# 2. Bar Chart
actual_counts = [list(y_test).count(l) for l in labels]
predicted_counts = [list(y_pred).count(l) for l in labels]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, actual_counts, width, label='Actual Data', color='#3498db')
plt.bar(x + width/2, predicted_counts, width, label='AI Predictions', color='#2ecc71')

plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Actual Data vs. Model Predictions')
plt.xticks(x, labels)
plt.legend()
plt.savefig('sentiment_barchart.png') 
plt.close()

print("Visualizations saved as 'confusion_matrix.png' and 'sentiment_barchart.png'!")
# ==========================================

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