# -- Amortization table representation -- #

# Libraries to use
import numpy as np
from typing import List, Optional
import pandas as pd

# Local scripts
from Cashflow import Cashflow


# Define IRR function
def irr(investment: List[Cashflow]) -> float:
    cfs_dict = {cf.t: cf.amount for cf in investment}
    cfs = [cfs_dict.get(i, 0) for i in range(max(cfs_dict.keys()) + 1)]
    # Reverse the cashflow list to have the higher-order powers first
    solution = np.roots(cfs[::-1])
    # Select only the solutions that make sense
    # Criteria: only real-solutions
    #candidate_filter = (solution.imag == 0) & (solution.real > 0)
    real_solution = solution[solution.imag == 0].real
    # Get back the "rates": rate =  1 / sol - 1
    rates = 1 / real_solution - 1
    # Return the nearest to zero
    return np.min(rates[rates > 0]) if any(rates > 0) else np.max(rates)

# Now let's define the Amortization class
class Amortization:
    
    def __init__(self, amount: float, rate: float, n: int):
        self.amount = amount
        self.rate = rate
        self.n = n
        
    def get_annuity(self) -> float:
        return self.rate * self.amount / (1 - (1+self.rate)**(-self.n))
    
    def to_cashflows(self, t: Optional[int] = None) -> List[Cashflow]:
        if t is None:
            t = self.n
        if not (0 < t <= self.n):
            raise ValueError(f"Use 0 < t <= n {self.n}")
        annuity = self.get_annuity()
        return [
            Cashflow(amount=-self.amount, t=0),
            *[Cashflow(amount=annuity, t=i+1) for i in range(t)]
        ]
    
    def get_table(self) -> pd.DataFrame:
        """
        b: Balance
        a: Annuity
        t: Time
        p: Principal payment
        i: Interest payment
        """
        b = self.amount # Initial balance
        a = self.get_annuity() # Constant annuity
        rows = [{"t": 0, "balance": b}]
        for t in range(1, self.n + 1):
            i = self.rate * b
            p = a - i
            b = b - p
            rows.append(
                {
                    "t": t,
                    "principal": p,
                    "interest": i,
                    "annuity": a,
                    "balance": b
                }
            )
        return pd.DataFrame(rows)
    
    
    def get_enriched_table(self, prob_of_default: float, loss_given_default: float) -> pd.DataFrame:
        
        enriched_table = self.get_table().copy()
        
        irr_evo = [irr(self.to_cashflows(t=i)) for i in range(1, self.n+1)]
        irr_evo.insert(0, 0)
        
        default_evo = [(1-prob_of_default)**i * prob_of_default if i<self.n
                       else (1-prob_of_default)**i
                       for i in range(1, self.n+1)]
        
        default_evo.insert(0, prob_of_default)
        
        enriched_table['irr'] = irr_evo
        enriched_table['prob'] = default_evo
        enriched_table['exp_loss'] = enriched_table['balance'] * prob_of_default * loss_given_default
        
        return enriched_table
    
    def expected_irr(self, prob_of_default: float, loss_given_default: float) -> float:
        
        table = self.get_enriched_table(prob_of_default=prob_of_default, loss_given_default=loss_given_default)
        
        total_irr = np.sum(table['irr'] * table['prob'])
        
        return round(total_irr, 4)