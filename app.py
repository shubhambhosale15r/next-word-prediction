from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Load the trained model and tokenizer
MODEL_DIR = 'model'  # Directory where the model and tokenizer are stored
MODEL_FILE = 'model.h5'  # Name of the model file
TOKENIZER_FILE = 'tokenizer.pkl'  # Name of the tokenizer file

# Ensure the model and tokenizer paths exist
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
tokenizer_path = os.path.join(MODEL_DIR, TOKENIZER_FILE)

if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    raise FileNotFoundError("Model or tokenizer file not found. Check file paths.")

model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Maximum length of sequences (update this based on your training settings)
MAX_LEN = 18

# Function to predict next words
def predict_next_words(model, tokenizer, text, max_len, top_n=3):
    tokenized_text = tokenizer.texts_to_sequences([text])[0]
    if not tokenized_text:  # Handle empty or invalid input
        return [{"error": "Input text contains no recognizable tokens."}]
    
    # Pad the sequence
    padded_token_text = pad_sequences([tokenized_text], maxlen=max_len, padding='pre')
    
    # Predict probabilities for the next word
    predictions = model.predict(padded_token_text, verbose=0)[0]
    
    # Get the top N predictions
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_words = [
        {"word": word, "probability": float(predictions[index])}
        for word, index in tokenizer.word_index.items() if index in top_indices
    ]
    return top_words

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html exists in the templates directory

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form.get('text', '').strip()
        if not input_text:
            return jsonify({"error": "Input text is required."}), 400
        try:
            predictions = predict_next_words(model, tokenizer, input_text, MAX_LEN, top_n=3)
            return jsonify({"predictions": predictions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
