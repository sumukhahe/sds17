import pickle
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
from helper_functions import logging


def training_pipeline(X_train, y_train):
    """
    Trains an XGBoost model and optionally saves it.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series or np.ndarray): Training data target labels.

    Returns:
        XGBClassifier: The trained XGBoost model object.
    """

    try:
        # Initialize the XGBoost classifier
        classification_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        # Train the model
        classification_model.fit(X_train, y_train)
        logging.info("Model trained successfully.")

        #  Optionally save the model (consider a model selection strategy in production)
        with open('best_classifier.pkl', 'wb') as f:
            pickle.dump(classification_model, f)
            logging.info("Model successfully pickled.")

        return classification_model

    except Exception as e:  # Catch generic exception for broader error handling
        logging.error(f"Error during training: {e}")
        raise  # Re-raise the exception for handling in the calling code


articact_path = "C:/Users/bpanda31/Downloads/DSC/MLOPs/demo_streamlit/risk_classification/scripts/best_classifier.pkl"


def load_model(path):
    """
    Loads a pickled model from the specified path.

    Args:
        path (str): Path to the pickled model file.

    Returns:
        object: The loaded model object.

    Raises:
        FileNotFoundError: If the model file is not found at the specified path.
    """

    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model

    except FileNotFoundError as e:
        logging.error(f"Model file not found: {path}")
        raise  # Re-raise the exception for handling in the calling code
  

def prediction_pipeline(X_val):
    """
    Makes predictions on the data using the loaded model and label encoder.

    Args:
      X_val (pd.DataFrame): Validation data features.

    Returns:
      np.ndarray: Array of predicted target labels.
    """
    try:
        # Load the model
        with open('best_classifier.pkl', 'rb') as file:
            model = pickle.load(file)

        # Make predictions
        predictions = model.predict(X_val)

        # Load the label encoder (assuming it's used for encoding target labels)
        with open('label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)
        predictions = label_encoder.inverse_transform(predictions)
        # If the model predicts class probabilities, decode them using the label encoder
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = label_encoder.inverse_transform(predictions.argmax(axis=1))            
        return predictions

    except FileNotFoundError as e:
        logging.error(f"Error loading model or label encoder: {e}")
        raise  # Re-raise the exception for handling in the calling code


def evaluation_matrices(X_val, y_val):
    """
    Calculates and logs evaluation metrics for the model.

    Args:
      X_val (pd.DataFrame): Validation data features.
      y_val (pd.Series or np.ndarray): Validation data target labels.

    Returns:
      tuple: A tuple containing the confusion matrix, accuracy score, and classification report.

    Raises:
      FileNotFoundError: If the label encoder file is not found.
    """

    try:
        # Load the label encoder
        with open('label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)

        # Predictions assuming prediction_pipeline returns categorical labels
        pred_vals = prediction_pipeline(X_val)   

        # Decode y_val using the loaded LabelEncoder
        decoded_y_vals = label_encoder.inverse_transform(y_val)
        
        # Calculate the confusion matrix with actual decoded labels
        conf_matrix = confusion_matrix(decoded_y_vals, pred_vals, labels=label_encoder.classes_)
        
        # Additional evaluation metrics
        acc_score = accuracy_score(decoded_y_vals, pred_vals)
        class_report = classification_report(decoded_y_vals, pred_vals)

        # Log evaluation metrics
        logging.info("Confusion Matrix:\n%s", pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))
        logging.info("Accuracy Score: %f", acc_score)
        logging.info("Classification Report:\n%s", class_report)

        return conf_matrix, acc_score, class_report

    except FileNotFoundError:
        logging.error("Label encoder file not found: label_encoder.pkl")
        raise  # Re-raise the exception for handling in the calling code

