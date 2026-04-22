from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# --- Load the Model and Vectorizer ---
with open('/home/cybersecurity/training/week6/week6-repo/day2/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('/home/cybersecurity/training/week6/week6-repo/day2/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['review']
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)