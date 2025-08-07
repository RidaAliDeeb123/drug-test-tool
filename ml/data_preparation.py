import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from loguru import logger

class DrugResponseDataPreparer:
    def create_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        data = {
            'gender': np.random.choice(['male', 'female'], n_samples),
            'age': np.random.normal(50, 15, n_samples).astype(int),
            'dosage': np.random.uniform(0.1, 100, n_samples),
            'response': np.random.binomial(1, 0.5, n_samples)
        }
        return pd.DataFrame(data)

    def prepare_training_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            X = df[['gender', 'age', 'dosage']]
            y = df['response']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return {}