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
# 1. Clone the repository
git clone https://github.com/AlhassenSabeeh/bitcoin_predictor.git
cd bitcoin_predictor

# 2. Create and activate virtual environment
python -m venv myenv
source myenv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Update data and run the application
python update_data.py
python app.py

# after run python update_date.py

🚀 BITCOIN PREDICTOR - DATA UPDATE TOOL
============================================================
🕐 Update started at: 2025-11-16 20:33:31

🔍 Checking system dependencies...
✅ Basic dependencies check passed
==================================================
📊 WIKIPEDIA SENTIMENT DATA UPDATE
==================================================
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
Device set to use cpu
✅ Sentiment analysis pipeline loaded successfully
🚀 Starting Wikipedia sentiment analysis pipeline...
📥 Fetching Wikipedia Bitcoin page edits...
✅ Fetched 18069 Wikipedia revisions
🧠 Analyzing sentiment of Wikipedia edits...
📊 Found 16738 valid sentiment scores out of 18069 revisions
✅ Analyzed sentiment for 2799 days with 16738 valid scores
✅ Sentiment analysis complete. File saved as 'wikipedia_edits.csv'
✅ Wikipedia data update completed successfully
   - Data points: 5746
   - Date range: 2010-02-23 00:00:00 to 2025-11-16 00:00:00
==================================================
🤖 BITCOIN PREDICTION MODEL UPDATE
==================================================
🚀 Starting Bitcoin prediction pipeline...
📊 Loading Bitcoin price data from Yahoo Finance...
✅ Loaded 4079 days of Bitcoin data (up to 2025-11-16 00:00:00)
🔄 Merging price data with Wikipedia sentiment...
✅ Loaded sentiment data for 5746 days
✅ Successfully merged price and sentiment data
⚙️  Creating technical features...
✅ Created 15 features across 4 time horizons
🤖 Training XGBoost model...
📈 Training on 4079 samples with 15 features...
✅ Model training complete and saved
🎯 Prediction: DOWN (Confidence: 61.15%)
✅ Prediction pipeline complete!
✅ Bitcoin model update completed successfully
   - Features used: 15
   - Training date: 2025-11-16T20:55:50.637579

============================================================
📋 UPDATE SUMMARY
============================================================
🕐 Started: 2025-11-16 20:33:31
🕐 Finished: 2025-11-16 20:55:50
⏱️  Duration: 1338.87 seconds

🎯 PREDICTION RESULTS:
   Next day prediction: DOWN
   Confidence: 61.15%
   Current Price: $94,127.77
   UP Probability: 38.849998474121094%
   DOWN Probability: 61.150001525878906%
   Model Training: 2025-11-16

✅ Update process completed!
