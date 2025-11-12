# Bitcoin Price Predictor 🔮

A machine learning web application that predicts Bitcoin price movements using Wikipedia sentiment analysis and technical indicators.

## Features
- Real-time Bitcoin price predictions
- Wikipedia edit sentiment analysis
- Automated daily model updates
- Web interface for easy access
- Confidence scoring

## How It Works
1. Analyzes Wikipedia Bitcoin page edits for sentiment
2. Processes Bitcoin price data and technical indicators
3. Uses XGBoost machine learning model
4. Predicts next-day price movement (UP/DOWN)

## Local Development
```bash
git clone https://github.com/AlhassenSabeeh/bitcoin_predictor.git
cd bitcoin_predictor
pip install -r requirements.txt
python update_data.py
python app.py