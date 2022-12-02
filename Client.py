# -- Client representation -- #

# Libraries to use
from typing import Dict
from sklearn.linear_model import LogisticRegression

# Local scripts
from Woe import WoeMapper

class Client:
    
    def __init__(
        self,
        customer_age: str,
        months_at_address: str,
        residence_status: str,
        employment: str,
        income: str,
        months_with_bank: str,
        other_credits: str,
        balance: str
    ):
        self.customer_age = customer_age
        self.months_at_address = months_at_address
        self.residence_status = residence_status
        self.employment = employment
        self.income = income
        self.months_with_bank = months_with_bank
        self.other_credits = other_credits
        self.balance = balance
        
    def to_dict(self) -> Dict:
        return self.__dict__
        
    def update_attribute(self, name: str, value: str):
        valid_attributes = list(self.to_dict().keys())
        if name not in valid_attributes:
            raise ValueError(f"Attribute {name} not found in {valid_attributes}")
        setattr(self, name, value)
        return self
        
    def get_prob_of_default(self, model: LogisticRegression, woe_mapper: WoeMapper) -> float:
        df = woe_mapper.transform(self.to_dict())
        prob_of_default, _ = model.predict_proba(df).tolist().pop()
        return prob_of_default