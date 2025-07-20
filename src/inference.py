from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from utils import load_model, load_digits_data


def run_inference(model, X, y_true=None):
    """
    Run inference using the trained model
    """
    print("Running inference...")
    predictions = model.predict(X)
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: {predictions.min()} to {predictions.max()}")
    
    if y_true is not None:
        accuracy = accuracy_score(y_true, predictions)
        print(f"Inference accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, predictions))
    
    return predictions


def main():
    """
    Main inference function
    """
    print("Loading trained model...")
    try:
        model = load_model("model_train.pkl")
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Error: model_train.pkl not found. Please run training first.")
        return
    
    print("Loading digits dataset...")
    X, y = load_digits_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run inference
    predictions = run_inference(model, X, y)
    
    # Show some sample predictions
    print("\nSample predictions (first 10):")
    for i in range(min(10, len(predictions))):
        print(f"Sample {i}: Predicted {predictions[i]}, Actual {y[i]}")


if __name__ == "__main__":
    main() 