import itertools
import pandas as pd
import scipy.stats as ss
import numpy as np

# https://stackoverflow.com/questions/46498455/categorical-features-correlation
def cramers_v(confusion_matrix):
    """ 
    Calculate Cramers V statistic for categorial-categorial association.
    Uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
        
    Parameters
    ----------
    confusion_matrix: np.array
        2D matrix representation of cross-tabulation table between two categorical variables
    Returns
    -------
    cramers v: float
        Returns the Cramers V statistic for categorial-categorial association    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def crammers_correlation_matrix(dataframe, features):
    """ 
    Calculate Cramers V correlation matrix for all features in dataframe
        
    Parameters
    ----------
    dataframe: pandas Dataframe
        dataframe with data of interest
    features: np.array
        categorical features of intereset from dataframe
    Returns
    -------
    cramers v: np.array
        Returns a symetric matrix with correlation between features of interest
    """    
    features = list(features)
    matrix = -1*np.ones((len(features), len(features)))
    for i, feat_i in enumerate(features):
        for j, feat_j in enumerate(features):
            symetric = matrix[j][i]
            if symetric >= 0:
                continue

            confusion_matrix = pd.crosstab(dataframe[feat_i], dataframe[feat_j]).values
            corr = cramers_v(confusion_matrix)
            matrix[i][j] = corr
            matrix[j][i] = corr
            
    return matrix