import json
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_config(config_path="config/config.json"):
    """
    Load configuration from JSON file
    """
    # Handle relative path from src directory
    import os
    if not os.path.exists(config_path):
        # Try going up one directory
        config_path = os.path.join("..", config_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_digits_data():
    """
    Load the digits dataset from sklearn
    """
    digits = load_digits()
    return digits.data, digits.target


def save_model(model, filepath="model_train.pkl"):
    """
    Save the trained model using joblib
    """
    joblib.dump(model, filepath)


def load_model(filepath="model_train.pkl"):
    """
    Load the trained model using joblib
    """
    return joblib.load(filepath) 