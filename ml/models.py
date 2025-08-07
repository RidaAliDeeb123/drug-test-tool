import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any
import joblib
from pathlib import Path
from loguru import logger

class DrugResponsePredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.encoder = LabelEncoder()

    def train(self, X: pd.DataFrame, y: pd.Series):
        try:
            X_encoded = X.copy()
            if 'gender' in X_encoded.columns:
                X_encoded['gender'] = self.encoder.fit_transform(X_encoded['gender'])
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_encoded, y)
            joblib.dump(self.model, self.model_dir / "model.joblib")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.model:
            raise ValueError("Model not trained")
            
        X_encoded = X.copy()
        if 'gender' in X_encoded.columns:
            X_encoded['gender'] = self.encoder.transform(X_encoded['gender'])
            
        return pd.Series(self.model.predict(X_encoded))