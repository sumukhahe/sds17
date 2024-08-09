from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from helper_functions import logging
import pickle
import pandas as pd
import numpy as np


def create_data_pipeline(data):
    """
    Creates a data processing pipeline for categorical and numerical features identified from the data types.

    This pipeline includes OneHotEncoding for categorical features with the first category dropped
    and MinMax scaling for numerical features.

    Args:
        data (pd.DataFrame): The pandas DataFrame containing the data.

    Returns:
        Pipeline: The created data processing pipeline, or None if no features are found.
    """

    categorical_features = []
    numerical_features = []

    # Check if data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        logging.error("Input data must be a pandas DataFrame.")
        return None

    # Get data types and separate features
    data_types = data.dtypes
    for col, dtype in data_types.items():
        if pd.api.types.is_categorical_dtype(dtype):
            categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            numerical_features.append(col)
        else:
            logging.warning(f"Column '{col}' has data type '{dtype}'. Ignoring for pipeline creation.")

    # Check if any features were found
    if not categorical_features and not numerical_features:
        logging.error("No categorical or numerical features found in the data. Pipeline creation failed.")
        return None

    # Create column transformer and pipeline
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features),
            ('num', MinMaxScaler(), numerical_features)
        ])

    pipeline = Pipeline([
        ('col_transformer', column_transformer)
    ])

    logging.info('Data pipeline created successfully:')
    return pipeline


def save_pipeline(pipeline, filename):
    """
    Saves the machine learning pipeline to a file.

    Args:
      pipeline (object): The machine learning pipeline to save.
      filename (str): The name of the file to save the pipeline to.

    Raises:
      ValueError: If the filename is empty or not a string.
    """

    if not isinstance(filename, str) or not filename:
        raise ValueError("Filename must be a non-empty string.")

    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)

    logging.info(f'Pipeline saved to: {filename}')


def load_pipeline(filename):
    """
    Loads a machine learning pipeline from a file.

    Args:
      filename (str): The name of the file containing the pipeline.

    Returns:
      object: The loaded machine learning pipeline.

    Raises:
      FileNotFoundError: If the specified file is not found.
    """

    if not isinstance(filename, str) or not filename:
        raise ValueError("Filename must be a non-empty string.")

    try:
        with open(filename, 'rb') as f:
            pipeline = pickle.load(f)
        logging.info(f'Pipeline loaded from: {filename}')
        return pipeline
    except FileNotFoundError:
        raise FileNotFoundError(f"Pipeline file not found: {filename}")


def encode_response_variable(y):
    """
    Encodes the response variable (y) using label encoding.

    Args:
        y (pd.Series or np.ndarray): The response variable data.

    Returns:
        np.ndarray: The encoded response variable.

    Raises:
        ValueError: If the input data (y) is not a pandas Series or NumPy array.
    """

    try:
        # Check data type
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("Input data (y) must be a pandas Series or NumPy array.")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Saving the label encoder for later use in decoding predictions (optional)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

        logging.info('Labels encoded for the response variable:')
        return y_encoded

    except ValueError as e:
        logging.error(f"Error encoding response variable: {e}")
        raise  # Re-raise the exception for handling in the calling code
    
       
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
      X (pd.DataFrame): The features data.
      y (pd.Series or np.ndarray): The target labels.
      test_size (float, optional): Proportion of data for the testing set. Defaults to 0.2.
      random_state (int, optional): Seed for random splitting. Defaults to 42.

    Returns:
      tuple: A tuple containing the training and testing data splits (X_train, X_test, y_train, y_test).

    Raises:
      ValueError: If the input data (X or y) is not a pandas DataFrame, Series, or NumPy array.
    """

    try:
        # Check data types
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)) or not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("Input data (X and y) must be pandas DataFrames, Series, or NumPy arrays.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info('Data is split into training and testing sets.')
        return X_train, X_test, y_train, y_test

    except ValueError as e:
        logging.error(f"Error splitting data: {e}")
        raise  # Re-raise the exception for handling in the calling code
