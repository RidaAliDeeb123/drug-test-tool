import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from typing_extensions import TypedDict
from loguru import logger

# Constants
CACHE_TTL = timedelta(days=7)
API_BASE_URL = "https://api.fda.gov/drug/label.json"
DEFAULT_LIMIT = 100

@dataclass
class DrugMetadata:
    manufacturer: str
    approval_date: str
    indications: List[str]
    contraindications: List[str]
    clinical_trials: List[str]
    pharmacology: Dict[str, Any]
    mechanism_of_action: List[str]

class GenderSpecificInfo(TypedDict):
    warnings: List[str]
    dose_adjustments: List[str]
    adverse_events: List[str]
    pharmacokinetics: List[str]
    pharmacodynamics: List[str]
    clinical_studies: List[str]
    risk_factors: List[str]
    pregnancy_category: Optional[str]
    lactation_risk: Optional[str]

class FDADrugData:
    def __init__(self, cache_dir: str = '.fda_cache'):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
        
        self.gender_patterns = {
            'female': [r'female[s]*', r'woman[s]*', r'girl[s]*', r'menstrual', r'pregnancy'],
            'male': [r'male[s]*', r'man[s]*', r'boy[s]*', r'testosterone', r'prostate']
        }
        
        self.study_bias_patterns = {
            'male_only': [r'male subjects only', r'excluded female patients'],
            'gender_mismatch': [r'gender differences not evaluated']
        }

    def _load_cache(self) -> None:
        cache_file = self._cache_dir / 'fda_cache.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
            except Exception as e:
                logger.error(f"Cache load failed: {e}")

    def search_drug(self, drug_name: str, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
        cache_key = f"search_{drug_name}_{limit}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]['data']
            
        try:
            params = {'search': f'openfda.brand_name:{drug_name}', 'limit': limit}
            response = requests.get(API_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get('results', [])
            
            if not results:
                params = {'search': f'openfda.generic_name:{drug_name}', 'limit': limit}
                response = requests.get(API_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                results = response.json().get('results', [])
            
            self._cache[cache_key] = {'data': results, 'timestamp': datetime.now()}
            self._save_cache()
            return results
            
        except Exception as e:
            logger.error(f"FDA API request failed: {e}")
            return []

    def get_gender_specific_info(self, drug_data: Dict[str, Any]) -> GenderSpecificInfo:
        gender_info: GenderSpecificInfo = {
            'warnings': [], 'dose_adjustments': [], 'adverse_events': [],
            'pharmacokinetics': [], 'pharmacodynamics': [], 'clinical_studies': [],
            'risk_factors': [], 'pregnancy_category': None, 'lactation_risk': None
        }
        
        for section in ['warnings', 'dosage_and_administration']:
            if section in drug_data:
                content = drug_data[section]
                if isinstance(content, str):
                    content = [content]
                
                for item in content:
                    item_str = str(item).lower()
                    if any(re.search(p, item_str) for p in self.gender_patterns['female']):
                        gender_info['warnings'].append(item)
        
        return gender_info

    def create_drug_profile(self, drug_name: str) -> Optional[Dict[str, Any]]:
        try:
            drug_data = self.search_drug(drug_name)
            if not drug_data:
                return None
                
            drug_info = drug_data[0]
            gender_info = self.get_gender_specific_info(drug_info)
            
            return {
                'drug_name': drug_name,
                'gender_specific_info': gender_info,
                'approval_date': drug_info.get('approval_date'),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Profile creation failed: {e}")
            return None