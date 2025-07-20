import pytest
import json
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config, load_digits_data
from train import train_model


class TestConfiguration:
    """Test configuration file loading"""
    
    def test_config_file_exists(self):
        """Test that configuration file exists"""
        assert os.path.exists("config/config.json")
    
    def test_config_loads_successfully(self):
        """Test that configuration file loads successfully"""
        config = load_config()
        assert isinstance(config, dict)
        assert len(config) > 0
    
    def test_required_hyperparameters_exist(self):
        """Test that all required hyperparameters exist in configuration"""
        config = load_config()
        required_params = ['C', 'solver', 'max_iter', 'random_state']
        
        for param in required_params:
            assert param in config, f"Missing required parameter: {param}"
    
    def test_hyperparameter_data_types(self):
        """Test that hyperparameters have correct data types"""
        config = load_config()
        
        assert isinstance(config['C'], (int, float)), "C should be numeric"
        assert isinstance(config['solver'], str), "solver should be string"
        assert isinstance(config['max_iter'], int), "max_iter should be integer"
        assert isinstance(config['random_state'], int), "random_state should be integer"
    
    def test_hyperparameter_values(self):
        """Test that hyperparameter values are reasonable"""
        config = load_config()
        
        assert config['C'] > 0, "C should be positive"
        assert config['solver'] in ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], "Invalid solver"
        assert config['max_iter'] > 0, "max_iter should be positive"
        assert config['random_state'] >= 0, "random_state should be non-negative"


class TestModelCreation:
    """Test model creation and training"""
    
    def test_train_model_returns_logistic_regression(self):
        """Test that train_model function returns a LogisticRegression object"""
        config = load_config()
        X, y = load_digits_data()
        
        model = train_model(X, y, config)
        assert isinstance(model, LogisticRegression)
    
    def test_model_is_fitted(self):
        """Test that the model has been fitted (has coefficients)"""
        config = load_config()
        X, y = load_digits_data()
        
        model = train_model(X, y, config)
        assert hasattr(model, 'coef_'), "Model should have coefficients after fitting"
        assert hasattr(model, 'classes_'), "Model should have classes after fitting"
        assert model.coef_ is not None, "Model coefficients should not be None"
        assert model.classes_ is not None, "Model classes should not be None"


class TestModelAccuracy:
    """Test model accuracy and performance"""
    
    def test_model_accuracy_threshold(self):
        """Test that model accuracy is above a reasonable threshold"""
        config = load_config()
        X, y = load_digits_data()
        
        model = train_model(X, y, config)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # The digits dataset is relatively easy, so we expect high accuracy
        assert accuracy > 0.8, f"Model accuracy {accuracy:.4f} is below threshold 0.8"
        print(f"Model accuracy: {accuracy:.4f}")
    
    def test_model_predictions_shape(self):
        """Test that model predictions have correct shape"""
        config = load_config()
        X, y = load_digits_data()
        
        model = train_model(X, y, config)
        y_pred = model.predict(X)
        
        assert y_pred.shape == y.shape, "Predictions should have same shape as targets"
    
    def test_model_predictions_range(self):
        """Test that model predictions are within valid range"""
        config = load_config()
        X, y = load_digits_data()
        
        model = train_model(X, y, config)
        y_pred = model.predict(X)
        
        # For digits dataset, predictions should be integers 0-9
        assert all(pred in range(10) for pred in y_pred), "All predictions should be digits 0-9"


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_digits_data_loading(self):
        """Test that digits dataset loads correctly"""
        X, y = load_digits_data()
        
        assert X.shape[1] == 64, "Features should be 64-dimensional (8x8 flattened)"
        assert len(X) == len(y), "Number of samples should match number of targets"
        assert all(target in range(10) for target in y), "All targets should be digits 0-9" 