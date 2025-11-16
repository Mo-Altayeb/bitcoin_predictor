import pickle
import pandas as pd
import os
import json
from datetime import datetime
import numpy as np

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
        """Enhanced feature importance with better feature name extraction"""
        if self.model is None:
            if not self.load_model():
                return self.get_sample_feature_importance()
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
                
                # Try multiple ways to get feature names
                feature_names = self._get_feature_names()
                
                if len(feature_names) == len(importance_scores):
                    # Combine and sort by importance
                    features = list(zip(feature_names, importance_scores))
                    features.sort(key=lambda x: x[1], reverse=True)
                    
                    # Return top features
                    top_features = features[:15]  # Increased limit
                    
                    return {
                        "features": [f[0] for f in top_features],
                        "importance": [float(f[1]) for f in top_features],
                        "total_features": len(features),
                        "categories": self._categorize_features([f[0] for f in top_features]),
                        "top_feature": top_features[0][0] if top_features else None
                    }
                else:
                    print(f"⚠️ Feature name mismatch: {len(feature_names)} names vs {len(importance_scores)} importance scores")
                    return self.get_sample_feature_importance()
            
            return self.get_sample_feature_importance()
            
        except Exception as e:
            print(f"❌ Error getting feature importance: {e}")
            return self.get_sample_feature_importance()
    
    def _get_feature_names(self):
        """Get feature names from multiple possible sources"""
        feature_names = []
        
        # Try feature_info.json first
        if os.path.exists(self.feature_info_path):
            try:
                with open(self.feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                if 'predictors' in feature_info:
                    feature_names = feature_info['predictors']
                    print(f"✅ Loaded {len(feature_names)} feature names from feature_info.json")
                    return feature_names
            except Exception as e:
                print(f"⚠️ Error loading feature info: {e}")
        
        # Try model attributes
        if hasattr(self.model, 'feature_names_in_'):
            feature_names = self.model.feature_names_in_.tolist()
        elif hasattr(self.model, '_feature_names'):
            feature_names = self.model._feature_names
        elif hasattr(self.model, 'get_booster'):
            try:
                feature_names = self.model.get_booster().feature_names
            except:
                pass
        
        # If still no features, use default feature set
        if not feature_names:
            feature_names = self._get_default_feature_names()
            print("⚠️ Using default feature names")
        
        return feature_names
    
    def _get_default_feature_names(self):
        """Return the standard feature names used in the model"""
        return [
            'close', 'sentiment', 'neg_sentiment', 
            'close_ratio_2', 'trend_2', 'edit_2',
            'close_ratio_7', 'trend_7', 'edit_7',
            'close_ratio_60', 'trend_60', 'edit_60',
            'close_ratio_365', 'trend_365', 'edit_365'
        ]
    
    def _categorize_features(self, features):
        """Enhanced feature categorization"""
        categories = {
            "price": [],
            "sentiment": [],
            "wikipedia": [],
            "technical": []
        }
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(term in feature_lower for term in ['close_ratio', 'price', 'close']):
                categories["price"].append(feature)
            elif any(term in feature_lower for term in ['sentiment']):
                categories["sentiment"].append(feature)
            elif any(term in feature_lower for term in ['edit']):
                categories["wikipedia"].append(feature)
            elif any(term in feature_lower for term in ['trend', 'momentum', 'volume', 'rsi', 'volatility']):
                categories["technical"].append(feature)
            else:
                # Default to technical for unrecognized features
                categories["technical"].append(feature)
        
        print(f"📊 Categorized features: Price({len(categories['price'])}), Sentiment({len(categories['sentiment'])}), Wikipedia({len(categories['wikipedia'])}), Technical({len(categories['technical'])})")
        return categories
    
    def get_sample_feature_importance(self):
        """Provide realistic sample feature importance data"""
        print("📝 Generating sample feature importance data...")
        
        sample_features = [
            'close_ratio_7', 'sentiment', 'close_ratio_60', 'trend_7',
            'neg_sentiment', 'close_ratio_365', 'edit_7', 'trend_60',
            'close_ratio_2', 'edit_60', 'trend_365', 'edit_365',
            'trend_2', 'edit_2', 'close'
        ]
        
        # Generate realistic importance scores (sum to 1)
        base_importance = np.random.dirichlet(np.ones(len(sample_features))).tolist()
        # Sort by importance
        features_sorted = sorted(zip(sample_features, base_importance), 
                               key=lambda x: x[1], reverse=True)
        
        features = [f[0] for f in features_sorted]
        importance = [float(f[1]) for f in features_sorted]
        
        return {
            "features": features,
            "importance": importance,
            "total_features": len(features),
            "categories": self._categorize_features(features),
            "top_feature": features[0] if features else None,
            "is_sample_data": True
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