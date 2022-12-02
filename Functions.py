# -- General functions script -- #

# Libraries to use
from scipy.optimize import minimize_scalar
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Local scripts
from Amortization import Amortization

# Let's generate a simple function in order to generate a quick eda plots
def get_eda_plots(data: pd.DataFrame,
                  feature: str,
                  is_discrete: bool=False):
    
    """
    Generate a simple EDA plot charts for continous and discrete variables
    """  
    
    # First discrimination
    if is_discrete==False: # Continous variable
        
        # MCT statistics.
        stats = pd.DataFrame(data[feature].describe()).T
        stats['median'] = data[feature].median()
        stats['mode'] = float(data[feature].mode()[0])
        
        # Plot histogram and boxplot
        fig, ax = plt.subplots(1, 2, figsize=(14,4), sharex=True)
        fig.suptitle(f'Distribution for: {feature}', fontsize=15)

        sns.boxplot(x=data[feature], ax=ax[1], color='#A7A698').set(title='Boxplot');
        sns.histplot(data[feature], ax=ax[0], stat='percent',
                     color='#A7A698', discrete=False).set(title='Probability Distribution');
    
        return stats
    
    else: # Discrete variable
        
        # Some frequency measures
        freq = {i: [round(list(data[feature]).count(i) / len(data), 4)] for i in list(data[feature].unique())}
        freq = pd.DataFrame.from_dict(freq)
        freq.index = ['frequency']
        
        # Plot the histogram.
        plt.figure(figsize=(5,4))
        sns.histplot(data[feature], stat='percent', color='#A7A698',
                     discrete=is_discrete).set(title='Probability Distribution');
        
        return freq

# Let's define a simple search for our optimal rate. The idea is to minimize an error function
def search_optimal_rate(
        amortization_amount: float,
        amortization_periods: int,
        prob_of_default: float,
        loss_given_default: float,
        min_rate: float,
        max_rate: float,
        target_expected_irr: float = 0
) -> float:
    
    # Define a generalized function in order to get the expected IRR for a given amortization structure
    get_expected = lambda amount, rate, n: Amortization(amount=amount, 
                                                        rate=rate, 
                                                        n=n).expected_irr(prob_of_default=prob_of_default,
                                                                          loss_given_default=loss_given_default)
    
    # Define the error function to be minimize
    def error(rate):
        
        error = (target_expected_irr - get_expected(amount=amortization_amount,
                                                    rate=rate,
                                                    n=amortization_periods))**2
        return error
    
    # Minimize process
    min_error = minimize_scalar(fun=error, bounds=(min_rate, max_rate), method='bounded')
    
    return min_error.x