# -- CashFlow representation class -- #

# Utility: Cashflow class 
class Cashflow:
    
    def __init__(self, amount: float, t: int):
        self.amount = amount
        self.t = t

    def __repr__(self) -> str:
        return f"Cashflow({self.amount}, {self.t})"
        
    def present_value(self, r: float) -> 'Cashflow':
        pv_amount = self.amount * (1 + r) ** (-self.t)
        return Cashflow(amount=pv_amount, t=0)
