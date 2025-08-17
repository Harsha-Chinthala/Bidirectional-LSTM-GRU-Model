from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from gensim.models import Word2Vec
import pickle
import numpy as np
import string
from rapidfuzz import process

app = Flask(__name__)

# -------------------- Custom Keras Layer -------------------- #
@register_keras_serializable()
class SumAlongAxis(keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# -------------------- Load Models and Data -------------------- #
try:
    model = keras.models.load_model(
        'amazon_sentiment_model.keras',
        custom_objects={'SumAlongAxis': SumAlongAxis}
    )
    if not model:
        raise ValueError("Model failed to load")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    wv_model = Word2Vec.load('word2vec_model.bin')

    with open('stop_words.pkl', 'rb') as f:
        stop_words = pickle.load(f)

except Exception as e:
    print(f"Error loading models or data: {e}")
    model = None
    wv_model = None
    stop_words = []

# -------------------- Preprocessing Functions -------------------- #
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stop_words(words, stop_words):
    return [w for w in words if w.lower() not in stop_words]

# -------------------- Sentiment Prediction -------------------- #
def predict_sentiment(comment):
    if not model or not wv_model:
        return 0.5  # Neutral if models not loaded

    comment = remove_punctuation(comment)
    words = remove_stop_words(comment.split(), stop_words)
    word_set = set(wv_model.wv.index_to_key)
    valid_words = [w for w in words if w in word_set]

    if not valid_words:
        return 0.5

    X = np.zeros((1, 25, 100))
    nw = 24
    for w in reversed(valid_words):
        if nw < 0:
            break
        if w in word_set:
            X[0, nw] = wv_model.wv[w]
            nw -= 1

    prediction = model.predict(X, verbose=0)
    return float(prediction[0][0])

# -------------------- Routes -------------------- #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        comment = data.get('comment', '')

        sentiment_score = predict_sentiment(comment)

        if sentiment_score >= 0.6:
            label = 'Positive'
        elif sentiment_score <= 0.4:
            label = 'Negative'
        else:
            label = 'Neutral'

        return jsonify({'label': label})

    except Exception as e:
        return jsonify({'error': f'Sorry, something went wrong: {str(e)}'}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        data = request.get_json(force=True)
        question = data.get('question', '')

        # Predefined answers
        answers = {
            "project name": "Sentiment Analysis of Product Reviews for Enhanced Customer Feedback which is a hybrid model of LSTM and Bidirectional GRU.",
            "description": "This project analyzes customer reviews and classifies them into positive, negative, or neutral sentiments using deep learning.",
            "dataset": "We use Amazon product reviews for training the model.",
            "LSTM":"LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture used in deep learning for sequence prediction tasks.",
            "Bidirectional GRU":"A Bidirectional GRU processes input sequences in both forward and backward directions.",
            "technologies": "Flask, TensorFlow, Word2Vec, Python",
            "goal": "Provide businesses with insights into customer feedback using sentiment analysis.",
            "sentiment analysis": "Categorizes comments as positive, negative, or neutral.",
            "preprocessing": "Remove punctuation, stop words, and vectorize text using Word2Vec.",
            "word2vec": "Converts words into vectors capturing their meaning.",
            "model used": "Deep learning model with embedding layers and GRU/LSTM layers.",
            "training": "Trained on a large set of Amazon reviews.",
            "model performance": "Evaluated using accuracy, precision, and recall.",
            "applications": "Helps businesses analyze customer feedback and improve products.",
            "improvements": "Train on larger datasets or add aspect-based sentiment analysis.",
            "how model works": "Preprocess text → Word2Vec vectorization → deep learning model → output sentiment score.",
            "model accuracy": "Achieved 90.03% accuracy on the test dataset.",
        }

        # Variants for fuzzy matching
        question_variants = {
            "project name": ["project title", "project name", "title of the project"],
            "description": ["description", "project about"],
            "LSTM":["LSTM"],
            "Bidirectional GRU":["Bidirectional GRU"],
            "dataset": ["dataset", "what dataset"],
            "technologies": ["technologies", "used technologies"],
            "model accuracy": ["accuracy", "model accuracy"],
            "how model works": ["how model works", "working of model"],
            "preprocessing": ["preprocessing", "data preprocessing"],
        }

        # Fuzzy matching
        best_match_key = None
        best_score = 0
        for key, variants in question_variants.items():
            for variant in variants:
                score = process.extractOne(question, [variant])[1]
                if score > best_score:
                    best_score = score
                    best_match_key = key

        if best_match_key and best_score >= 70:
            answer = answers.get(best_match_key, "I don't have an answer for that.")
        else:
            answer = "Sorry, I couldn't understand your question."

        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': f'Sorry, something went wrong: {str(e)}'}), 500

# -------------------- Run App -------------------- #
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)