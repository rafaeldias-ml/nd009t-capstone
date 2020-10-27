import pickle
import numpy as np
import pandas as pd
from functools import partial
from collections import Counter

from category_encoders import BinaryEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from utils import pickle_obj, unpickle_obj

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

## Data conversion functions
def to_int(value, default=-1):
    """
    Converts a value to integer
    ---
    Arguments
        value: string
            Value to be converted
        default: int
            Default value when conversion is not possible
    Returns
        value: int
            Converted value
    """
    if value is None:
        return default

    try:
        value = int(round(float(value)))
        value = default if np.isnan(value) else value
    except:
        value = default
        
    return value

def to_float(value, default=-1):
    """
    Converts a value to float
    ---
    Arguments
        value: string
            Value to be converted
        default: int
            Default value when conversion is not possible
    Returns
        value: float
            Converted value
    """    
    if value is None:
        return default
    
    try:
        value = float(value)
        value = default if np.isnan(value) else value
    except:
        value = default
        
    return value


## Preprocessing functions 
# These functions have a standard signature  fn(value, arg1, arg2, default) 
# to make the application easier based on the data dictionary definition
def bounded(value, arg1=1, arg2=1e9, default=-1):
    """
    Ensure a value is bounded to min and max value, 
    returning default value otherwise
    ---
    Arguments
        value: string
            Value to be converted
        arg1: int
            Lower boundary
        arg2: int
            Upper boundary
        default: int
            Default value when conversion is not possible or value is
            outside boundaries
    Returns
        value: int
            Value constrained to the boundaries
    """
    value = to_int(value, default)

    low = int(arg1)
    high = int(arg2)

    return value if low <= value < high else default

def numeric(value, arg1=None, arg2=None, default=-1):
    """
    Converts to numeric value
    ---
    Arguments
        value: string
            Value to be converted
        arg1: None
            Not used
        arg2: None
            Not used
        default: int
            Default value when conversion is not possible
    Returns
        value: float
            Converted value
    """
    value = to_float(value, default)

    return value

def cut(value, arg1=0, arg2=None, default=-1):
    """
    Converts ordinal ranged values to sequential categories
    ---
    Arguments
        value: string
            Value to be converted
        arg1: string
            List of ordinal values separated by '-''
        arg2: None
            Not used
        default: int
            Default value when conversion is not possible
    Returns
        category: int
            Converted value
    """
    value = to_int(value, default)
    boundaries = arg1
    bounds = list(map(int, boundaries.split('-')))

    if value < bounds[0] or value > bounds[-1]:
        return int(default)
  
    for category, upper_boundary in enumerate(bounds):
        if upper_boundary > value:
            break
        else:
            category += 1

    return category

def label_encoding(value, arg1='a-b', arg2=None, default=np.nan):
    """
    Enforce a categorical value from a list of allowed ones
    ---
    Arguments
        value: string
            Value to be enforced
        arg1: string
            List of categorical values separated by '-''
        arg2: None
            Not used
        default: np.nan
            Default value when enforcement is not possible
    Returns
        value: string
            Enforced value
    """
    expected_values = set(arg1.split('-'))
    return value if value in expected_values else default

def data_pre_processing(dataframe, attributes_df):
    """
    Perfom pre-processing of dataframe, according to specifications
    of attributes dataframe (based on the data dictionary)
    During this phase, features are renamed with english names,
    validated and pre-processed according to the data type and expected 
    values described on the data dictionary.
    Finally, missing values are filled with -1 which is the typical
    indicator of unknow value in the data dictionary.
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataset dataframe with raw data
        attributes_df: pd.DataFrame
            Dataframe with feature attributes, new names and pre-processing instructions
            based on the data dictionary
    Returns
        dataframe: pd.DataFrame
            Dataset with pre-processed features
    """
    map_functions_dict = {'categorical': (bounded, int),
                          'ordinal': (bounded, int),
                          'numeric': (numeric, float),
                          'label_encoding': (label_encoding, str),
                          'cut_categorical': (cut, int),
                          'cut_ordinal': (cut, int),
                          'drop': (numeric, int)
                         }

    # restrict transformation dataframe to features available on the dataset
    transformation_df = attributes_df[attributes_df.Feature.isin(dataframe.columns.values)]

    # walk through attributes transformation dataset retrieving transformation
    # parameters for each feature and then apply the transformation to respective
    # column in the dataset
    for index, row in transformation_df.iterrows():
        if row.ProcessingFn in ['drop']:
            column = row.Feature
            dataframe = dataframe.drop(columns=[column])
            continue

        base_fn, return_type = map_functions_dict[row.ProcessingFn]
        proc_fn = partial(base_fn, arg1=row.Arg1, arg2=row.Arg2)
        dataframe[row.Feature] = dataframe[row.Feature].apply(proc_fn).astype(return_type)

    # finally, rename features with English names
    column_remaping = {row.Feature: row.Renaming for index, row in attributes_df.iterrows()}
    dataframe.rename(column_remaping, inplace=True, axis='columns')
    
    return dataframe.fillna(-1)


## Slicing routines
def get_numeric_dataframe(dataframe, attributes_df):
    """
    Returns a view of dataframe containing only numeric features
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataset dataframe with pre-processed data (i.e. renamed features)
        attributes_df: pd.DataFrame
            Dataframe with feature attributes, based on the data dictionary
    Returns
        dataframe: pd.DataFrame
            Dataset view with numeric features only
    """
    features_of_interest = set(attributes_df[attributes_df.ProcessingFn.isin(['numeric'])].Renaming)
    features_of_interest = features_of_interest.intersection(dataframe.columns.values)
    
    return dataframe.loc[:, features_of_interest]

def get_categorical_dataframe(dataframe, attributes_df):
    """
    Returns a view of dataframe containing only categorical features
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataset dataframe with pre-processed data (i.e. renamed features)
        attributes_df: pd.DataFrame
            Dataframe with feature attributes, based on the data dictionary
    Returns
        dataframe: pd.DataFrame
            Dataset view with categorical features only
    """
    cat_functions = ['categorical', 'label-encoding', 'cut_categorical']
    features_of_interest = set(attributes_df[attributes_df.ProcessingFn.isin(cat_functions)].Renaming)
    features_of_interest = features_of_interest.intersection(dataframe.columns.values)
    
    return dataframe.loc[:, features_of_interest]

def get_ordinal_dataframe(dataframe, attributes_df):
    """
    Returns a view of dataframe containing only ordinal features
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataset dataframe with pre-processed data (i.e. renamed features)
        attributes_df: pd.DataFrame
            Dataframe with feature attributes, based on the data dictionary
    Returns
        dataframe: pd.DataFrame
            Dataset view with ordinal features only
    """
    ord_functions = ['ordinal', 'cut_ordinal']
    features_of_interest = set(attributes_df[attributes_df.ProcessingFn.isin(ord_functions)].Renaming)
    features_of_interest = features_of_interest.intersection(dataframe.columns.values)
    
    return dataframe.loc[:, features_of_interest]

def get_low_and_high_cardinality_categorical_dfs(dataframe, attributes_df, threshold=5, fit=False):
    """
    Returns a tuple of dataframes containing categorical features only:
    - low cardinality: features with number of unique categories less than or equal to threshold
    - high cardinality: features with number of unique categories higher than threshold
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataset dataframe with pre-processed data (i.e. renamed features)
        attributes_df: pd.DataFrame
            Dataframe with feature attributes, based on the data dictionary
        threshold: int
            Threshold to consider high cardinality, based on number of categories 
        fit: boolean
            Indicates if we should measure cardinality or consider previously measured data
    Returns
        tuple: (pd.DataFrame, pd.DataFrame)
            (Dataset view with low cardinality features, Dataset view with high cardinality features)
    """
    # retrieve categorical features
    categorical_df = get_categorical_dataframe(dataframe, attributes_df)
    features = categorical_df.columns.values

    cardinality_count = {}
    # measure or read features cardinality
    if fit:
        for col in features:
            cardinality_count[col] = len(categorical_df[col].unique())
        
        pickle_obj(cardinality_count, 'cardinality_count')
    else:
        cardinality_count = unpickle_obj('cardinality_count')

    # split low and high cardinality features, based on threshold        
    high_cardinality_features = [feature for feature, cardinality in cardinality_count.items() if cardinality > threshold]
    low_cardinality_features = set(features)-set(high_cardinality_features)
    
    # create cardinality views
    low_cardinality_cat_df = categorical_df.loc[:, low_cardinality_features]
    high_cardinality_cat_df = categorical_df.loc[:, high_cardinality_features]
    
    return low_cardinality_cat_df, high_cardinality_cat_df


## Imputation routines
def numeric_values_imputation(numeric_dataframe, fit=False, nan_value=-1):
    """
    Perform imputation of missing values for numeric features using median.
    ---
    Arguments
        numeric_dataframe: pd.DataFrame
            Dataframe with pre-processed data (i.e. renamed features), numeric features only
        fit: boolean
            Indicates if we should train or load an imputer
        nan_value: Any
            Value to be considered as missing value
    Returns
        dataframe: pd.DataFrame
            Dataframe with missing values imputed
    """
    # Train or load a simple imputer, responsible for filling missing values with feature median
    if fit:
        imputer = SimpleImputer(missing_values=nan_value, strategy='median')
        imputer.fit(numeric_dataframe)
        
        pickle_obj(imputer, 'numeric_imputer')
    else:
        imputer = unpickle_obj('numeric_imputer')
    
    # input missing values
    transformed = imputer.transform(numeric_dataframe)

    # construct a dataframe from np.array values and original column names
    return pd.DataFrame(transformed, columns=numeric_dataframe.columns.values)

def categorical_values_imputation(categorical_dataframe, fit=False, nan_value=-1):
    """
    Perform imputation of missing values for categorical features using most frequent value.
    ---
    Arguments
        categorical_dataframe: pd.DataFrame
            Dataframe with pre-processed data (i.e. renamed features), categorical features only
        fit: boolean
            Indicates if we should train or load an imputer
        nan_value: Any
            Value to be considered as missing value
    Returns
        dataframe: pd.DataFrame
            Dataframe with missing values imputed
    """
    # Train or load a simple imputer, responsible for filling missing values with most frequent
    if fit:
        imputer = SimpleImputer(missing_values=nan_value, strategy='most_frequent')
        imputer.fit(categorical_dataframe)
        
        pickle_obj(imputer, 'categorical_imputer')
    else:
        imputer = unpickle_obj('categorical_imputer')

    # input missing values
    transformed = imputer.transform(categorical_dataframe)

    # construct a dataframe from np.array values and original column names    
    return pd.DataFrame(transformed, columns=categorical_dataframe.columns.values)

def ordinal_values_imputation(ordinal_dataframe, fit=False, nan_value=-1):
    """
    Perform imputation of missing values for ordinal features using median value.
    ---
    Arguments
        ordinal_dataframe: pd.DataFrame
            Dataframe with pre-processed data (i.e. renamed features), ordinal features only
        fit: boolean
            Indicates if we should train or load an imputer
        nan_value: Any
            Value to be considered as missing value
    Returns
        dataframe: pd.DataFrame
            Dataframe with missing values imputed
    """
    # Train or load a simple imputer, responsible for filling missing values with feature median
    if fit:
        imputer = SimpleImputer(missing_values=nan_value, strategy='median')
        imputer.fit(ordinal_dataframe)
        
        pickle_obj(imputer, 'ordinal_imputer')
    else:
        imputer = unpickle_obj('ordinal_imputer')

    # input missing values
    transformed = imputer.transform(ordinal_dataframe)

    # construct a dataframe from np.array values and original column names    
    return pd.DataFrame(transformed, columns=ordinal_dataframe.columns.values)

def values_imputation(dataframe, attributes_df, fit=False, nan_value=-1):
    """
    Umbrella function for imputation of missing values. This function rellies on
    feature type specific imputation routines such as numeric_values_imputation and
    ordinal_values_imputation.
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataframe with pre-processed data (i.e. renamed features)
        attributes_df: pd.DataFrame
            Dataframe with feature attributes, based on the data dictionary            
        fit: boolean
            Indicates if we should train or load an imputers
        nan_value: Any
            Value to be considered as missing value
    Returns
        dataframe: pd.DataFrame
            Dataframe with missing values imputed
    """
    # get numeric features view
    numeric_df = get_numeric_dataframe(dataframe, attributes_df)
    numeric_df = numeric_values_imputation(numeric_df, fit=fit, nan_value=nan_value)
    
    # get categorical features view
    categorical_df = get_categorical_dataframe(dataframe, attributes_df)
    categorical_df = categorical_values_imputation(categorical_df, fit=fit, nan_value=nan_value)
    
    # get ordinal features view
    ordinal_df = get_ordinal_dataframe(dataframe, attributes_df)
    ordinal_df = ordinal_values_imputation(ordinal_df, fit=fit, nan_value=nan_value)
    
    # create a new dataframe out of processed pieces
    df = pd.concat([numeric_df, categorical_df, ordinal_df], axis=1)

    return df

def most_common(data):
   """
    Returns the most common element in a series
    ---
    Arguments
        data: list
            Series of data
    Returns
        element: Any
            Most common element of data
    """
    count = Counter(data)
    return count.most_common(1)[0][0]


## Encoding routines
def encode_ordinal_df(dataframe, fit=False):
   """
    Encode ordinal features, preserving the notion of order and dropping invariant features
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataframe with pre-processed data (i.e. renamed features), ordinal features only
        fit: boolean
            Indicates if we should train or load an encoder
    Returns
        dataframe: pd.DataFrame
            Dataframe with encoded data
    """
    # Train or load an encoder    
    if fit:
        encoder = OrdinalEncoder(cols=dataframe.columns.values, drop_invariant=True)
        encoder.fit(dataframe)
        
        pickle_obj(encoder, 'ordinal_encoder')
    else:
        encoder = unpickle_obj('ordinal_encoder')

    # transform data
    return encoder.transform(dataframe)

def encode_low_cardinality_categorical_df(dataframe, fit=False):
   """
    Encode low cardinality categorical features using OneHot Encoding and dropping invariant features
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataframe with pre-processed data (i.e. renamed features), low card. categorical features only
        fit: boolean
            Indicates if we should train or load an encoder
    Returns
        dataframe: pd.DataFrame
            Dataframe with encoded data
    """
    # Train or load an encoder    
    if fit:
        encoder = OneHotEncoder(cols=dataframe.columns.values, drop_invariant=True)
        encoder.fit(dataframe)
        
        pickle_obj(encoder, 'low_card_categorical_encoder')
    else:
        encoder = unpickle_obj('low_card_categorical_encoder')

    # transform data
    return encoder.transform(dataframe)

def encode_high_cardinality_categorical_df(dataframe, fit=False):
   """
    Encode high cardinality categorical features using Binary Encoding and dropping invariant features
    In Binary Encoding, features are converted to a binary representation and binary digits are used as new
    features.
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataframe with pre-processed data (i.e. renamed features), high card. categorical features only
        fit: boolean
            Indicates if we should train or load an encoder
    Returns
        dataframe: pd.DataFrame
            Dataframe with encoded data
    """
    # Train or load an encoder    
    if fit:
        encoder = BinaryEncoder(cols=dataframe.columns.values, drop_invariant=True)
        encoder.fit(dataframe)
        
        pickle_obj(encoder, 'high_card_categorical_encoder')
    else:
        encoder = unpickle_obj('high_card_categorical_encoder')

    # transform data
    return encoder.transform(dataframe)

def encode_features(dataframe, attributes_df, fit=False):
    """
    Umbrella function for feature encoding. This function rellies on
    feature type specific encoding routines such as encode_ordinal_df and
    encode_high_cardinality_categorical_df.
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataframe with pre-processed data (i.e. renamed features)
        attributes_df: pd.DataFrame
            Dataframe with feature attributes, based on the data dictionary            
        fit: boolean
            Indicates if we should train or load an imputers
    Returns
        dataframe: pd.DataFrame
            Dataframe with encoded features
    """
    # get feature type specific views
    numeric_df = get_numeric_dataframe(dataframe, attributes_df)
    ordinal_df = get_ordinal_dataframe(dataframe, attributes_df)
    low_cardinality_cat_df, high_cardinality_cat_df = get_low_and_high_cardinality_categorical_dfs(dataframe, attributes_df, fit=fit)

    # encode features using type specific encoders
    ordinal_df = encode_ordinal_df(ordinal_df, fit)
    low_cardinality_cat_df = encode_low_cardinality_categorical_df(low_cardinality_cat_df, fit)
    high_cardinality_cat_df = encode_high_cardinality_categorical_df(high_cardinality_cat_df, fit)

    # create a new dataframe out of processed pieces
    df = pd.concat([numeric_df, ordinal_df, low_cardinality_cat_df, high_cardinality_cat_df], axis=1)

    return df


## Normalization / Standardization functions
def features_normalization(dataframe, fit=False):
   """
    Performs feature normalization using a MinMax scaler.
    After normalization, features values will be in the range [0:1]
    while preserving original distribution.
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataframe with encoded data
        fit: boolean
            Indicates if we should train or load a scaler
    Returns
        dataframe: pd.DataFrame
            Dataframe with scaled features
    """
    # Train or load a scaler
    if fit:
        scaler = MinMaxScaler()
        scaler.fit(dataframe)

        pickle_obj(scaler, 'minmax_scaler')
    else:
        scaler = unpickle_obj('minmax_scaler')

    # Transform data and recreate dataframe from np.array
    X = scaler.transform(dataframe)
    df = pd.DataFrame(X, columns=dataframe.columns)

    return df
    
def features_standardization(dataframe, fit=False):
   """
    Performs feature standardization using a Standard scaler.
    After standardization, features will have zero means and
    unit standard deviation, changing the original distribution.
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataframe with encoded data
        fit: boolean
            Indicates if we should train or load a scaler
    Returns
        dataframe: pd.DataFrame
            Dataframe with scaled features
    """
    # Train or load a scaler    
    if fit:
        scaler = StandardScaler()
        scaler.fit(dataframe)

        pickle_obj(scaler, 'standard_scaler')
    else:
        scaler = unpickle_obj('standard_scaler')

    # Transform data and recreate dataframe from np.array
    X = scaler.transform(dataframe)
    df = pd.DataFrame(X, columns=dataframe.columns)

    return df


## Utility functions to save features of interest
def save_feature_set(dataframe, attributes_df, label='features_of_interest', save_original_features=True):
  """
    Save list of features using their original or current names
    ---
    Arguments
        dataframe: pd.DataFrame
            Dataframe with pre-processed data
        attributes_df: pd.DataFrame
            Dataframe with feature attributes, based on the data dictionary
        label: string
            Filename for serialization
        save_original_features: boolean
            Flag indicating if we should save original or current feature names
    Returns
        None
    """
    # get current feature names
    renamed_features = set(dataframe.columns.values)
    # retrieve original feature names, using attributes dataframe
    original_features = attributes_df[attributes_df.Renaming.isin(renamed_features)].Feature.values
    
    # decide which feature set to save based on save_original_features flag
    features = original_features if save_original_features else renamed_features
    
    # serialize list of features
    pickle_obj(features, label)
    
def load_feature_set(label='features_of_interest'):
   """
    Load serialized feature set 
    ---
    Arguments
        label: string
            Filename
    Returns
        feature_set: List
            List of features deserialized
    """
    return unpickle_obj(label)