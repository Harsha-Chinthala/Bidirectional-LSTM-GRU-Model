from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from gensim.models import Word2Vec
import pickle
import numpy as np
import string
from tensorflow.keras.utils import register_keras_serializable
from rapidfuzz import process

app = Flask(__name__)

@register_keras_serializable()
class SumAlongAxis(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# Load models and other necessary files
try:
    model = tf.keras.models.load_model(
        'amazon_sentiment_model.keras',
        custom_objects={'SumAlongAxis': SumAlongAxis}
    )
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile after loading
    wv_model = Word2Vec.load('word2vec_model.bin')

    with open('stop_words.pkl', 'rb') as f:
        stop_words = pickle.load(f)
except Exception as e:
    print(f"Error loading models or data: {e}")

def remove_punctuation(s):
    """Remove punctuation from a string."""
    table = str.maketrans({key: None for key in string.punctuation})
    return s.translate(table)

def remove_stop_words(raw_sen, stop_words):
    """Remove stop words from a list of words."""
    return [w for w in raw_sen if w not in stop_words]

def predict_sentiment(comment):
    comment = remove_punctuation(comment)
    comment = remove_stop_words(comment.split(), stop_words)
    word_set = set(wv_model.wv.index_to_key)
    valid_words = [w for w in comment if w in word_set]

    if not valid_words:  # No recognizable words
        return 0.5  # Neutral score for random or non-meaningful input

    X = np.zeros((1, 25, 100))
    nw = 24
    for w in list(reversed(valid_words)):
        if w in word_set and nw >= 0:
            X[0, nw] = wv_model.wv[w]
            nw -= 1

    prediction = model.predict(X)
    return float(prediction[0][0])

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')
@app.route('/demo')
def demo():
    """Render the demo page."""
    return render_template('demo.html')
@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    try:
        data = request.get_json(force=True)
        comment = data['comment']
        
        # Get the sentiment score from the model
        sentiment_score = predict_sentiment(comment)

        # Classify sentiment based on the score
        if sentiment_score >= 0.6:
            label = 'Positive'
        elif sentiment_score <= 0.4:
            label = 'Negative'
        else:
            label = 'Neutral'  # Adjusted label for neutral sentiment

        # Return only the sentiment label
        return jsonify({'label': label})

    except Exception as e:
        return jsonify({'error': f'Sorry, something went wrong: {str(e)}'}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Handle the question answering functionality."""
    try:
        data = request.get_json(force=True)
        question = data['question']
        
        answers = {
            "project name": "Sentiment Analysis of Product Reviews for Enhanced Customer Feedback which is a hybrid model of LSTM and Bidirectional GRU.",
            "description": "This project analyzes customer reviews and classifies them into positive, negative, or neutral sentiments using deep learning.",
            "dataset": "We use Amazon product reviews for training the model.",
            "LSTM":"LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture used in deep learning for sequence prediction tasks, such as time series analysis, natural language processing, and speech recognition. LSTM is specifically designed to overcome the limitations of traditional RNNs, which struggle to retain information over long sequences due to the vanishing gradient problem.",
            "Bidirectional GRU":"A Bidirectional GRU (Gated Recurrent Unit) is an advanced type of recurrent neural network (RNN) that processes input sequences in both forward and backward directions, enhancing the model’s ability to capture context from both past and future states of a sequence.",
            "technologies": "This project uses Flask, TensorFlow, Word2Vec, and Python.",
            "goal": "The goal is to provide businesses with insights into customer feedback using sentiment analysis.",
            "sentiment analysis": "Sentiment analysis helps in understanding customer opinions by categorizing comments as positive, negative, or neutral.",
            "sentiment classification": "The model classifies sentiments into three categories: Positive, Negative, and Neutral.",
            "preprocessing": "We remove punctuation, stop words, and vectorize the text using Word2Vec.",
            "word2vec": "Word2Vec is a technique to convert words into vectors that capture their meaning.",
            "model used": "We are using a deep learning model built with TensorFlow. It combines embedding layers and GRU/LSTM layers.",
            "training": "The model was trained on a large set of Amazon reviews.",
            "model performance": "The model is evaluated using metrics like accuracy, precision, and recall.",
            "applications": "This project can help businesses analyze customer feedback and improve products or services.",
            "improvements": "Future improvements could involve training the model on a larger dataset or adding more advanced features like aspect-based sentiment analysis.",
            "how model works": "The model works by first preprocessing the customer review text, removing unnecessary punctuation and stop words. It then uses Word2Vec to convert the words into numerical vectors that capture semantic meanings. The processed input is passed through a deep learning model with embedding layers and GRU/LSTM layers. Finally, the model outputs a sentiment score, classifying the review as positive, negative, or neutral based on the probability of each sentiment.",
            "how accurate is the model": "The model achieved an accuracy of 90.3% on the test dataset, meaning it correctly predicted the sentiment of 85% of the reviews.",
            "how sentiment analysis works": "The sentiment analysis works by analyzing the words in the review and determining whether the overall sentiment is positive, negative, or neutral. It uses deep learning techniques and word embeddings to understand the context of the words in the review.",
            "model accuracy": "The model achieved an accuracy of 85% on the test dataset.",
            "precision": "The model's precision score is 0.82, meaning it correctly identifies positive reviews 82% of the time.",
            "recall": "The recall score is 0.80, which means the model captures 80% of all actual positive reviews.",
            "f1 score": "The F1 score of the model is 0.81, providing a balance between precision and recall.",
            "evaluation metrics": "The model was evaluated using accuracy, precision, recall, and F1 score. The model performs well with an accuracy of 85%.",
            "sentiment trends": "Positive sentiments dominate with 65%, while negative sentiments account for 20%, and neutral sentiments make up 15% of the analyzed reviews.",
            "performance evaluation": "The model's performance was evaluated using a test set of 10,000 Amazon reviews, and the accuracy reached 85%.",
        }

        question_variants = {
            "project name": ["What is the Project Title", "What is the Name of the Project", "What is the Title of the Project"],
            "description": ["What is the description of the project","What is the project about "],
            "LSTM":["What is LSTM"],
            "Bidirectional GRU":["what is Bidirectional GRU"],
            "evalution metrics":["what are the evalution metrics used"],
            "f1 score":["How much F1 score got","F1 Score","what is the F1 score"],
            "recall":["what is the Recall Value","Recall","Recall Value","How much Recall did you got","How much Recall"],
            "dataset": ["dataset", "what dataset", "which dataset", "what data is used", "dataset used"],
            "technologies": ["technologies", "what technologies", "which technologies", "what is used in this project"],
            "model accuracy": ["model accuracy", "how accurate is the model", "what is the accuracy of the model"],
            "model evaluation": ["model evaluation", "how is the model evaluated", "how was the model tested", "model testing"],
            "real-time prediction": ["real-time prediction", "how does the real-time prediction work", "real-time prediction process"],
            "preprocessing": ["data preprocessing", "how is the data preprocessed", "what preprocessing is done"],
            "training process": ["training process", "how was the model trained", "model training process"],
            "model building": ["model building", "how is the model built", "what is the architecture of the model"],
            "sentiment prediction": ["sentiment prediction", "how does sentiment prediction work", "sentiment analysis process"],
            "word2vec": ["word2vec", "what is word2vec", "how does word2vec work"],
            "feedback loop": ["feedback loop", "how does the feedback loop work", "how is the model improved"],
            "sentiment score": ["sentiment score", "how is the sentiment score calculated", "what is the sentiment score"],
            "post-processing": ["post-processing", "what is post-processing", "how is the output processed after prediction"],
            "testing and validation": ["testing and validation", "how was the model validated", "how is the model tested"],
            "model optimization": ["model optimization", "how is the model optimized", "what optimization techniques were used"],
            "accuracy vs loss": ["accuracy vs loss", "how is accuracy and loss tracked", "what is the relationship between accuracy and loss"],
            "deployment": ["deployment", "how is the model deployed", "how was the model deployed in the application"],
            "how model works": ["how does the model work", "how does the sentiment analysis model work", "what is the working of the model", "how the model predicts sentiment"],
            "how sentiment analysis works": ["how does the sentiment analysis work", "how sentiment analysis works", "what is sentiment analysis", "how is sentiment analysis done"]
        }

        # Find the best matching answer using fuzzy string matching
        best_match_key = None
        best_match_score = 0
        for key, variants in question_variants.items():
            for variant in variants:
                score = process.extractOne(question, [variant])[1]
                if score > best_match_score:
                    best_match_score = score
                    best_match_key = key

        # If a match is found and it meets the threshold, return the appropriate response
        if best_match_key and best_match_score >= 70:  # You can adjust the threshold
            answer =answers.get(best_match_key, "I'm sorry, I don't have an answer for that question.")
        else:
            answer = "Sorry, I couldn't understand your question."
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'Sorry, something went wrong: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)