import pandas as pd
from fhir.resources.bundle import Bundle
from fhir.resources.patient import Patient
from fhir.resources.medicationstatement import MedicationStatement
from typing import Union, Dict, Any, List
import json
from pathlib import Path
from loguru import logger


class DataIngestor:
    def __init__(self):
        self.standard_columns = {
            'required': ['patient_id', 'gender', 'birth_date'],
            'optional': ['dosage', 'medication_code']
        }

    def load_medical_data(self, source_type: str, source: Union[str, Dict]) -> pd.DataFrame:
        handlers = {
            'csv': self._handle_csv,
            'json': self._handle_json,
            'ehr': self._handle_ehr,
            'manual': self._handle_manual
        }

        if source_type not in handlers:
            raise ValueError(f"Unsupported source type: {source_type}")

        return handlers[source_type](source)

    def _handle_csv(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"CSV load failed: {e}")
            return pd.DataFrame()

    def _handle_json(self, file_path: str) -> pd.DataFrame:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # If the JSON file is a dictionary with a single list inside
                if isinstance(data, dict) and len(data) == 1 and isinstance(list(data.values())[0], list):
                    data = list(data.values())[0]

                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"JSON load failed: {e}")
            return pd.DataFrame()

    def _handle_ehr(self, fhir_data: Dict) -> pd.DataFrame:
        try:
            bundle = Bundle.parse_obj(fhir_data)
            records = []

            for entry in bundle.entry or []:
                if not entry.resource:
                    continue

                if isinstance(entry.resource, Patient):
                    records.append({
                        'patient_id': entry.resource.id,
                        'gender': entry.resource.gender,
                        'birth_date': entry.resource.birthDate
                    })

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"FHIR processing failed: {e}")
            return pd.DataFrame()

    def _handle_manual(self, data: Dict[str, Any]) -> pd.DataFrame:
        try:
            # Wrap in a list in case a single dictionary is passed
            return pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Manual data load failed: {e}")
            return pd.DataFrame()

    def detect_bias(self, df: pd.DataFrame) -> Dict[str, float]:
        if 'gender' not in df.columns:
            logger.warning("Gender column missing for bias detection.")
            return {}

        gender_counts = df['gender'].value_counts(normalize=True)
        return {
            'male_ratio': gender_counts.get('male', 0),
            'female_ratio': gender_counts.get('female', 0),
            'bias_score': abs(gender_counts.get('male', 0) - gender_counts.get('female', 0))
        }
