import pickle
import pandas as pd
import os
import json

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_path = 'models/saved_models/bitcoin_model.pkl'
        self.feature_info_path = 'models/saved_models/feature_info.json'
    
    def load_model(self):
        """Load the trained model and feature info"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Model loaded successfully")
                
                # Load feature information
                if os.path.exists(self.feature_info_path):
                    with open(self.feature_info_path, 'r') as f:
                        self.feature_info = json.load(f)
                    print("Feature information loaded")
                else:
                    self.feature_info = {}
                    
                return True
            else:
                print("No trained model found")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def model_exists(self):
        """Check if model file exists"""
        return os.path.exists(self.model_path)
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.model is None:
            if not self.load_model():
                return {"status": "no_model"}
        
        info = {
            "status": "loaded",
            "model_type": type(self.model).__name__,
            "features_used": getattr(self.model, 'feature_names_in_', []).tolist() if hasattr(self.model, 'feature_names_in_') else [],
            "n_features": len(getattr(self.model, 'feature_names_in_', [])),
            "feature_info": self.feature_info
        }
        
        return info
    
    def get_feature_importance(self):
        """Get feature importance from model"""
        if self.model is None:
            if not self.load_model():
                return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
                feature_names = getattr(self.model, '_feature_names_', [])
                
                if len(feature_names) == len(importance_scores):
                    # Combine and sort by importance
                    features = list(zip(feature_names, importance_scores))
                    features.sort(key=lambda x: x[1], reverse=True)
                    
                    # Return top 10 features
                    top_features = features[:10]
                    
                    return {
                        "features": [f[0] for f in top_features],
                        "importance": [float(f[1]) for f in top_features]
                    }
            
            return {}
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {}