import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
import pickle
import os
import json
from sklearn.metrics import precision_score, classification_report
from datetime import datetime

class BitcoinPricePredictor:
    def __init__(self):
        self.model = None
        self.predictors = []
        self.feature_names = []
    
    def load_data(self):
        """Load and prepare Bitcoin price data"""
        print("Loading Bitcoin price data...")
        
        try:
            btc_ticker = yf.Ticker("BTC-USD")
            btc = btc_ticker.history(period='max')
            
            if btc.empty:
                print("No Bitcoin data fetched from Yahoo Finance")
                return pd.DataFrame()
                
            btc = btc.reset_index()
            if 'Date' in btc.columns:
                btc['Date'] = btc['Date'].dt.tz_localize(None) if hasattr(btc['Date'].dt, 'tz_localize') else btc['Date']
            
            # Remove unnecessary columns if they exist
            for col in ["Dividends", "Stock Splits"]:
                if col in btc.columns:
                    del btc[col]
            
            # Standardize column names
            btc.columns = [c.lower() for c in btc.columns]
            
            print(f"Loaded {len(btc)} days of Bitcoin data")
            return btc
            
        except Exception as e:
            print(f"Error loading Bitcoin data: {e}")
            return pd.DataFrame()
    
    def merge_with_sentiment(self, btc_data):
        """Merge price data with sentiment data"""
        print("Merging with sentiment data...")
        
        if btc_data.empty:
            print("No Bitcoin data to merge")
            return pd.DataFrame()
        
        try:
            # Load sentiment data
            if not os.path.exists("wikipedia_edits.csv"):
                print("Sentiment data file not found")
                # Create empty sentiment columns
                btc_data['sentiment'] = 0
                btc_data['neg_sentiment'] = 0
                btc_data['edit_count'] = 0
                return btc_data
                
            bit_sent = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
            
            # Prepare dates for merging
            btc_data['date'] = pd.to_datetime(btc_data['date']).dt.normalize()
            
            # Merge datasets
            btc_data = btc_data.merge(bit_sent, left_on='date', right_index=True, how='left')
            
            # Fill missing sentiment values
            sentiment_cols = ['sentiment', 'neg_sentiment', 'edit_count']
            for col in sentiment_cols:
                if col in btc_data.columns:
                    btc_data[col] = btc_data[col].fillna(0)
                else:
                    btc_data[col] = 0
            
            btc_data = btc_data.set_index('date')
            return btc_data
            
        except Exception as e:
            print(f"Error merging sentiment data: {e}")
            # Add default sentiment columns
            btc_data['sentiment'] = 0
            btc_data['neg_sentiment'] = 0
            btc_data['edit_count'] = 0
            if 'date' in btc_data.columns:
                btc_data = btc_data.set_index('date')
            return btc_data
    
    def create_features(self, data):
        """Create technical features for prediction - MUST MATCH APP.PY"""
        print("Creating features...")
        
        if data.empty:
            print("No data for feature creation")
            return data
        
        # Create target variable
        data["tomorrow"] = data["close"].shift(-1)
        data["target"] = (data["tomorrow"] > data["close"]).astype(int)
        data = data.dropna(subset=['target'])
        
        horizons = [2, 7, 60, 365]
        new_predictors = ["close", "sentiment", "neg_sentiment"]

        for horizon in horizons:
            try:
                # Close ratio - MUST MATCH APP.PY
                rolling_close = data["close"].rolling(horizon, min_periods=1).mean()
                ratio_column = f"close_ratio_{horizon}"
                data[ratio_column] = data["close"] / rolling_close

                # Edit count - MUST MATCH APP.PY
                if 'edit_count' in data.columns:
                    rolling_edits = data["edit_count"].rolling(horizon, min_periods=1).mean()
                else:
                    rolling_edits = pd.Series(0, index=data.index)
                edit_column = f"edit_{horizon}"
                data[edit_column] = rolling_edits

                # Trend - MUST MATCH APP.PY EXACTLY
                trend_column = f"trend_{horizon}"
                data[trend_column] = (data["close"] > data["close"].shift(1)).rolling(horizon, min_periods=1).mean().fillna(0)

                new_predictors.extend([ratio_column, trend_column, edit_column])
                
            except Exception as e:
                print(f"Error creating features for horizon {horizon}: {e}")
                # Add default values for failed features
                data[f"close_ratio_{horizon}"] = 1.0
                data[f"edit_{horizon}"] = 0
                data[f"trend_{horizon}"] = 0.5

        self.predictors = new_predictors
        self.feature_names = new_predictors  # Store for feature importance
        
        print(f"Created {len(new_predictors)} features")
        return data
    
    def train_model(self, data):
        """Train the prediction model"""
        print("Training model...")
        
        if data.empty:
            print("No data for training")
            return None
        
        if len(self.predictors) == 0:
            print("No predictors available for training")
            return None
        
        try:
            # Use all data for training (no train/test split for production)
            train_data = data.copy()
            
            # Ensure all predictors exist
            for predictor in self.predictors:
                if predictor not in train_data.columns:
                    print(f"Warning: Predictor {predictor} not found, adding zeros")
                    train_data[predictor] = 0
            
            # Train XGBoost model
            self.model = XGBClassifier(
                random_state=1, 
                learning_rate=0.1, 
                n_estimators=100,
                eval_metric='logloss'
            )
            
            print(f"Training on {len(train_data)} samples with {len(self.predictors)} features...")
            self.model.fit(train_data[self.predictors], train_data["target"])
            
            # Store feature names in the model instance (XGBoost compatible way)
            self.model._feature_names = self.predictors
            
            # Save model
            os.makedirs('models/saved_models', exist_ok=True)
            model_path = 'models/saved_models/bitcoin_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save feature information
            feature_info = {
                'predictors': self.predictors,
                'feature_names': self.feature_names,
                'training_date': datetime.now().isoformat(),
                'training_samples': len(train_data)
            }
            
            with open('models/saved_models/feature_info.json', 'w') as f:
                json.dump(feature_info, f, indent=2)
            
            print("Model training complete and saved.")
            return self.model
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    def predict_next_day(self, data):
        """Predict next day price movement"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        if data.empty:
            raise ValueError("No data available for prediction")
        
        try:
            # Get the latest data point
            latest_data = data.iloc[-1:].copy()
            
            # Ensure all predictors are present
            for pred in self.predictors:
                if pred not in latest_data.columns:
                    print(f"Warning: Predictor {pred} not found, using 0")
                    latest_data[pred] = 0
            
            prediction = self.model.predict(latest_data[self.predictors])
            prediction_proba = self.model.predict_proba(latest_data[self.predictors])
            
            confidence = float(np.max(prediction_proba[0]))
            
            result = {
                "prediction": "UP" if prediction[0] == 1 else "DOWN",
                "confidence": round(confidence * 100, 2),
                "current_price": float(round(latest_data['close'].iloc[0], 2)),
                "prediction_proba": {
                    "up_probability": float(round(prediction_proba[0][1] * 100, 2)),
                    "down_probability": float(round(prediction_proba[0][0] * 100, 2))
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            # Return a safe default prediction
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {
                    "up_probability": 50.0,
                    "down_probability": 50.0
                }
            }
    
    def get_feature_importance(self):
        """Get feature importance for analytics"""
        if self.model is None:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_.tolist()
                feature_names = getattr(self.model, '_feature_names', self.predictors)
                
                # Combine and sort by importance
                features = list(zip(feature_names, importance_scores))
                features.sort(key=lambda x: x[1], reverse=True)
                
                # Return top features
                top_features = features[:10]
                
                return {
                    "features": [f[0] for f in top_features],
                    "importance": [float(f[1]) for f in top_features]
                }
            else:
                return {}
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {}
    
    def run_full_pipeline(self):
        """Run the complete prediction pipeline"""
        print("Starting Bitcoin prediction pipeline...")
        
        # Step 1: Load data
        btc_data = self.load_data()
        if btc_data.empty:
            print("Pipeline failed: No Bitcoin data loaded")
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {
                    "up_probability": 50.0,
                    "down_probability": 50.0
                }
            }
        
        # Step 2: Merge with sentiment
        merged_data = self.merge_with_sentiment(btc_data)
        if merged_data.empty:
            print("Pipeline failed: No merged data")
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {
                    "up_probability": 50.0,
                    "down_probability": 50.0
                }
            }
        
        # Step 3: Create features
        enhanced_data = self.create_features(merged_data)
        if enhanced_data.empty:
            print("Pipeline failed: No enhanced data")
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {
                    "up_probability": 50.0,
                    "down_probability": 50.0
                }
            }
        
        # Step 4: Train model
        model = self.train_model(enhanced_data)
        if model is None:
            print("Pipeline failed: Model training failed")
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {
                    "up_probability": 50.0,
                    "down_probability": 50.0
                }
            }
        
        # Step 5: Make prediction
        prediction = self.predict_next_day(enhanced_data)
        
        print("Prediction pipeline complete!")
        return prediction