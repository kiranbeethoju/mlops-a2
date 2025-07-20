from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import load_config, load_digits_data, save_model


def train_model(X, y, config):
    """
    Train a LogisticRegression model with given configuration
    """
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=config['random_state']
    )
    model.fit(X, y)
    return model


def main():
    """
    Main training function
    """
    print("Loading configuration...")
    config = load_config()
    print(f"Configuration loaded: {config}")
    
    print("Loading digits dataset...")
    X, y = load_digits_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("Training LogisticRegression model...")
    model = train_model(X, y, config)
    
    # Evaluate the model
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Training accuracy: {accuracy:.4f}")
    
    print("Saving model...")
    save_model(model, "model_train.pkl")
    print("Model saved as model_train.pkl")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))


if __name__ == "__main__":
    main() 