import pickle
import pandas as pd
import os
import json
from datetime import datetime

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_path = 'models/saved_models/bitcoin_model.pkl'
        self.feature_info_path = 'models/saved_models/feature_info.json'
        self.model_loaded_time = None
        self.feature_info = {}  # Initialize feature_info here
    
    def load_model(self):
        """Load the trained model and feature info"""
        try:
            if os.path.exists(self.model_path):
                print("🔄 Loading trained model...")
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_loaded_time = datetime.now()
                print("✅ Model loaded successfully")
                
                # Load feature information
                if os.path.exists(self.feature_info_path):
                    with open(self.feature_info_path, 'r') as f:
                        self.feature_info = json.load(f)
                    print("✅ Feature information loaded")
                else:
                    self.feature_info = {}
                    print("⚠️  No feature information found")
                    
                return True
            else:
                print("❌ No trained model found at", self.model_path)
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def model_exists(self):
        """Check if model file exists"""
        exists = os.path.exists(self.model_path)
        if exists:
            print("✅ Model file exists")
        else:
            print("❌ Model file not found")
        return exists
    
    def get_model_info(self):
        """Get comprehensive information about the trained model"""
        if self.model is None:
            if not self.load_model():
                return {
                    "status": "no_model",
                    "message": "No trained model available"
                }
        
        info = {
            "status": "loaded",
            "model_type": type(self.model).__name__,
            "model_loaded_time": self.model_loaded_time.isoformat() if self.model_loaded_time else "Unknown",
            "features_used": getattr(self.model, 'feature_names_in_', []).tolist() if hasattr(self.model, 'feature_names_in_') else [],
            "n_features": len(getattr(self.model, 'feature_names_in_', [])),
            "feature_info": self.feature_info,  # This is now safely initialized
            "model_parameters": {
                "n_estimators": getattr(self.model, 'n_estimators', 'Unknown'),
                "learning_rate": getattr(self.model, 'learning_rate', 'Unknown')
            } if hasattr(self.model, 'n_estimators') else {}
        }
        
        return info
    
    def get_feature_importance(self):
        """Get feature importance from model"""
        if self.model is None:
            if not self.load_model():
                return {
                    "error": "Model not loaded",
                    "features": [],
                    "importance": []
                }
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
                feature_names = getattr(self.model, '_feature_names', [])
                
                if not feature_names and hasattr(self.model, 'feature_names_in_'):
                    feature_names = self.model.feature_names_in_.tolist()
                
                if len(feature_names) == len(importance_scores):
                    # Combine and sort by importance
                    features = list(zip(feature_names, importance_scores))
                    features.sort(key=lambda x: x[1], reverse=True)
                    
                    # Return top 10 features
                    top_features = features[:10]
                    
                    return {
                        "features": [f[0] for f in top_features],
                        "importance": [float(f[1]) for f in top_features],
                        "total_features": len(features)
                    }
            
            return {
                "error": "No feature importance available",
                "features": [],
                "importance": []
            }
        except Exception as e:
            print(f"❌ Error getting feature importance: {e}")
            return {
                "error": str(e),
                "features": [],
                "importance": []
            }
    
    def get_model_freshness(self):
        """Calculate how fresh the model is"""
        if not self.model_exists():
            return "no_model"
        
        try:
            if os.path.exists(self.feature_info_path):
                with open(self.feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                
                training_date = datetime.fromisoformat(feature_info.get('training_date', '2020-01-01'))
                days_old = (datetime.now() - training_date).days
                
                if days_old < 1:
                    return "very_fresh"
                elif days_old < 7:
                    return "fresh"
                elif days_old < 30:
                    return "moderate"
                else:
                    return "stale"
            else:
                return "unknown"
                
        except Exception as e:
            print(f"⚠️  Error calculating model freshness: {e}")
            return "unknown"