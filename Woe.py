# -- WOE Mapper class -- #

# Libraries to use
import pandas as pd
from typing import List, Dict, Optional, Union

# Local scripts
import Binning as bn

# Class definition
class WoeMapper:
    
    def __init__(self, features: List[str]):
        self.features = features
        self.mapper = {}
        
    def fit(self, data: pd.DataFrame) -> 'WoeMapper':
        df = data[self.features]
        for feature in list(df.columns):
            self.mapper[feature] = {row[feature]: row["woe"] for _, row in bn.get_woe(data, feature).iterrows()}
        return self
            
    def transform(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        if not self.mapper:
            raise ValueError("Unfitted mapper detected.")
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        out = pd.DataFrame([])
        for feature in self.features:
            out[feature] = [self.mapper[feature][val] for val in data[feature].values]
        return pd.DataFrame(
            {
                feature: [self.mapper[feature][val] for val in data[feature].values]
                for feature in self.features
            }
        )
