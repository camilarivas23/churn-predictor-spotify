"""
Model Loading and Inference Module
Handles model predictions and feature importance extraction.
"""
import json
import joblib
import pandas as pd
from typing import Dict, List, Any, Tuple


class ChurnPredictor:
    """Load and use trained churn prediction model."""
    
    def __init__(self, model_path: str, features_path: str):
        """Initialize predictor with trained model and features.
        
        Args:
            model_path: Path to model.pkl (trained Random Forest)
            features_path: Path to features.json (feature names and order)
        """
        self.model_path = model_path
        self.features_path = features_path
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load features metadata
        with open(features_path, 'r') as f:
            self.features_metadata = json.load(f)
        
        self.feature_names = self.features_metadata.get('features', [])
        self.model_type = self.features_metadata.get('model_type', 'RandomForest')
        self.churn_threshold = self.features_metadata.get('churn_threshold', 0.5)
    
    def predict(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """Predict churn for a user given their features.
        
        Args:
            features_dict: Dictionary mapping feature names to values
            
        Returns:
            Dictionary with prediction, probability, and top contributing features
        """
        # Prepare feature array in correct order
        X = []
        for feat in self.feature_names:
            value = features_dict.get(feat, 0.0)
            X.append(value)
        
        X = pd.DataFrame([X], columns=self.feature_names)
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        prediction_proba = self.model.predict_proba(X)[0]
        
        # Churn probability is the probability of class 1 (CHURNED)
        churn_probability = float(prediction_proba[1])
        
        # Determine prediction label
        churn_label = "CHURNED" if prediction == 1 else "ACTIVE"
        
        # Get feature importance from the model
        feature_importances = self.model.feature_importances_
        
        # Get top contributing features
        top_features = self._get_top_features(
            feature_importances, 
            features_dict,
            top_k=3
        )
        
        return {
            "prediction": churn_label,
            "churn_probability": churn_probability,
            "confidence": float(max(prediction_proba)),
            "top_features": top_features,
            "all_feature_importances": [
                {"name": feat, "importance": float(imp)}
                for feat, imp in zip(self.feature_names, feature_importances)
            ]
        }
    
    def _get_top_features(self, importances: List[float], features_dict: Dict[str, float], 
                         top_k: int = 3) -> List[Dict[str, Any]]:
        """Get top contributing features with their values and importance.
        
        Args:
            importances: Feature importance array
            features_dict: Feature values dictionary
            top_k: Number of top features to return
            
        Returns:
            List of top features with importance and value
        """
        feature_importance_pairs = list(zip(self.feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_features = []
        for feat_name, importance in feature_importance_pairs[:top_k]:
            top_features.append({
                "name": feat_name,
                "importance": float(importance),
                "value": float(features_dict.get(feat_name, 0.0))
            })
        
        return top_features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in model order."""
        return self.feature_names
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get metadata about the trained model."""
        return {
            "model_type": self.model_type,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "n_estimators": getattr(self.model, 'n_estimators', None),
            "max_depth": getattr(self.model, 'max_depth', None),
            "churn_threshold": self.churn_threshold
        }


def load_model(model_path: str, features_path: str) -> ChurnPredictor:
    """Load a trained churn prediction model.
    
    Args:
        model_path: Path to model.pkl
        features_path: Path to features.json
        
    Returns:
        ChurnPredictor instance
    """
    return ChurnPredictor(model_path, features_path)


if __name__ == "__main__":
    # Example usage
    predictor = load_model("model.pkl", "features.json")
    
    # Create sample feature input
    sample_features = {
        "days_since_last_listen": 15,
        "listening_events_last_7d": 5,
        "avg_days_between_listens": 2.5,
        "is_active_last_7d": 1,
        "total_artist_listens": 150,
        "unique_artists_90d": 25,
        "listen_to_artist_ratio": 0.15,
        "repeat_listen_rate": 0.3,
        "artist_popularity_avg": 65,
        "track_energy_avg": 0.7,
        "track_danceability_avg": 0.6,
        "is_high_engagement_user": 1,
        "account_age_days": 365,
        "avg_track_tempo": 120
    }
    
    prediction = predictor.predict(sample_features)
    print(json.dumps(prediction, indent=2))
