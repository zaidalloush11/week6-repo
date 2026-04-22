import pickle
import re

# 1. We must use the exact same cleaning function from Day 2
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower() 
    text = re.sub(r'[^a-z\s]', '', text) 
    return text.strip()

# 2. "Turn on" (Load) the saved files
print("Loading AI model and vocabulary...")
try:
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    print("Files loaded successfully!\n")
except FileNotFoundError:
    print("Error: Could not find the .pkl files. Make sure you are running this in the right folder!")
    exit()

# 3. Live Testing Loop

print("   LIVE AI SENTIMENT TESTER   ")


while True:
    user_input = input("Type a review (or type 'quit' to exit): ")
    
    if user_input.lower() == 'quit':
        print("Shutting down AI...")
        break
        
    # Clean -> Convert to Numbers -> Predict
    cleaned = clean_text(user_input)
    vectorized = loaded_vectorizer.transform([cleaned])
    prediction = loaded_model.predict(vectorized)[0]
    
    print(f"AI Prediction: {prediction}\n")