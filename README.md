# Bidirectional LSTM-GRU Model for Sentiment Analysis

This project implements a Bidirectional LSTM-GRU model for sentiment analysis using TensorFlow and Flask.

## Project Structure

```
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── amazon_sentiment_model.keras    # Trained model file
├── word2vec_model.bin             # Word2Vec model file
├── stop_words.pkl                 # Stop words pickle file
├── templates/                     # HTML templates
│   ├── index.html                # Home page
│   ├── demo.html                 # Demo page
│   └── test.html                 # Test page
├── venv_py311/                   # Python 3.11 virtual environment
└── README.md                     # This file
```

## Setup Instructions

### Prerequisites
- Python 3.11 (required for gensim compatibility)
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Bidirectional-LSTM-GRU-Model
   ```

2. **Create virtual environment with Python 3.11:**
   ```bash
   python3.11 -m venv venv_py311
   source venv_py311/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install rapidfuzz
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   - Open your browser and go to `http://localhost:5000`
   - The application will be available on port 5000

## Features

- **Sentiment Analysis**: Analyze text and classify sentiment as Positive, Negative, or Neutral
- **Interactive Demo**: Web interface for testing sentiment analysis
- **Question Answering**: Chat-like interface to ask questions about the project
- **Real-time Processing**: Instant sentiment prediction

## Model Information

- **Architecture**: Bidirectional LSTM-GRU hybrid model
- **Accuracy**: 90.03% on test dataset
- **Dataset**: Amazon product reviews
- **Word Embeddings**: Word2Vec for text vectorization

## API Endpoints

- `GET /` - Home page
- `GET /demo` - Demo page
- `POST /predict` - Sentiment prediction API
- `POST /ask_question` - Question answering API

## Deployment

This project is ready for deployment on platforms like:
- Heroku
- AWS
- Google Cloud Platform
- Railway
- Render

Make sure to set the appropriate environment variables and use a production WSGI server like Gunicorn.

## Technologies Used

- **Backend**: Flask, Python
- **Machine Learning**: TensorFlow, Keras, Gensim
- **Text Processing**: NLTK, Word2Vec
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Gunicorn (recommended)

## License

This project is for educational purposes. 