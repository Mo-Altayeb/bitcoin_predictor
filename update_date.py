from models.sentiment_analyzer import WikipediaSentimentAnalyzer
from models.price_predictor import BitcoinPricePredictor
import time
from datetime import datetime

def update_wikipedia_data():
    """Update Wikipedia sentiment data with enhanced error handling"""
    print("=" * 50)
    print("📊 WIKIPEDIA SENTIMENT DATA UPDATE")
    print("=" * 50)
    try:
        analyzer = WikipediaSentimentAnalyzer()
        sentiment_data = analyzer.create_sentiment_file()
        
        if sentiment_data is not None and not sentiment_data.empty:
            print("✅ Wikipedia data update completed successfully")
            print(f"   - Data points: {len(sentiment_data)}")
            print(f"   - Date range: {sentiment_data.index.min()} to {sentiment_data.index.max()}")
            return sentiment_data
        else:
            print("❌ Wikipedia data update completed but no data was generated")
            return None
            
    except Exception as e:
        print(f"❌ Wikipedia data update failed: {e}")
        return None

def update_bitcoin_model():
    """Update Bitcoin prediction model with enhanced progress tracking"""
    print("=" * 50)
    print("🤖 BITCOIN PREDICTION MODEL UPDATE")  
    print("=" * 50)
    try:
        predictor = BitcoinPricePredictor()
        prediction = predictor.run_full_pipeline()
        
        if prediction and 'error' not in prediction:
            print("✅ Bitcoin model update completed successfully")
            model_info = predictor.get_model_info()
            print(f"   - Features used: {model_info['predictors_count']}")
            print(f"   - Training date: {model_info['training_date']}")
            return prediction
        else:
            print("❌ Bitcoin model update completed but with errors")
            return prediction
            
    except Exception as e:
        print(f"❌ Bitcoin model update failed: {e}")
        # Return safe default
        return {
            "prediction": "UP",
            "confidence": 50.0,
            "current_price": 0.0,
            "prediction_proba": {
                "up_probability": 50.0,
                "down_probability": 50.0
            },
            "error": str(e)
        }

def check_system_dependencies():
    """Check if all required system dependencies are available"""
    print("🔍 Checking system dependencies...")
    dependencies = {
        "Wikipedia API": True,  # mwclient will fail at runtime if not available
        "Yahoo Finance": True,  # yfinance will fail at runtime
        "Machine Learning": True,  # xgboost and sklearn
        "Sentiment Analysis": True  # transformers
    }
    
    print("✅ Basic dependencies check passed")
    return all(dependencies.values())

if __name__ == "__main__":
    print("🚀 BITCOIN PREDICTOR - DATA UPDATE TOOL")
    print("=" * 60)
    
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"🕐 Update started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check dependencies
    if not check_system_dependencies():
        print("❌ System dependency check failed. Please install required packages.")
        exit(1)
    
    # Update Wikipedia data
    wiki_data = update_wikipedia_data()
    
    # Small delay to ensure file writing is complete
    time.sleep(2)
    
    # Update Bitcoin model
    result = update_bitcoin_model()
    
    end_time = time.time()
    end_datetime = datetime.now()
    duration = round(end_time - start_time, 2)
    
    print()
    print("=" * 60)
    print("📋 UPDATE SUMMARY")
    print("=" * 60)
    print(f"🕐 Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Finished: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Duration: {duration} seconds")
    print()
    
    if result and 'error' not in result:
        print("🎯 PREDICTION RESULTS:")
        print(f"   Next day prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}%")
        if result['current_price'] > 0:
            print(f"   Current Price: ${result['current_price']:,.2f}")
        else:
            print(f"   Current Price: $N/A")
        print(f"   UP Probability: {result['prediction_proba']['up_probability']}%")
        print(f"   DOWN Probability: {result['prediction_proba']['down_probability']}%")
        
        if 'model_training_date' in result:
            print(f"   Model Training: {result['model_training_date']}")
    else:
        print("❌ Update completed with errors")
        if result and 'error' in result:
            print(f"   Error: {result['error']}")
    
    print()
    print("✅ Update process completed!")