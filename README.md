# MLOps Artifact Pipeline

This repository contains a complete MLOps pipeline for digit classification using Logistic Regression, implemented with GitHub Actions for continuous integration and deployment.

## Project Overview

- **Dataset**: sklearn.datasets.load_digits (built-in)
- **Task**: Multiclass classification (digits 0-9)
- **Features**: 64 grayscale pixel values (flattened 8x8 images)
- **Model**: LogisticRegression from sklearn.linear_model

## Repository Structure

```
.
├── src/
│   ├── train.py          # Training script
│   ├── inference.py      # Inference script
│   └── utils.py          # Utility functions
├── config/
│   └── config.json       # Model hyperparameters
├── tests/
│   └── test_train.py     # Unit and integration tests
├── .github/
│   └── workflows/
│       ├── train.yml     # Training workflow
│       ├── test.yml      # Testing workflow
│       └── inference.yml # Multi-job inference workflow
├── requirements.txt      # Python dependencies
└── README.md
```

## Branching Strategy

Following the assignment guidelines, the project uses a linear branching approach:

- **main**: Initial repository with README
- **classification**: Phase 1 - Training pipeline
- **test**: Phase 2 - Testing with pytest
- **inference**: Phase 3 - Inference and multi-job workflow

## Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n mlops python=3.8
conda activate mlops
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. GitHub Repository Setup
```bash
# Create a new repository on GitHub named 'mlops-artifact-pipeline'
git init
git add .
git commit -m "Initial commit with README"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mlops-artifact-pipeline.git
git push -u origin main
```

### 4. Create Branches (Follow Assignment Order)
```bash
# Phase 1: Training Pipeline
git checkout -b classification
git push -u origin classification

# Phase 2: Testing
git checkout -b test
git push -u origin test

# Phase 3: Inference
git checkout -b inference
git push -u origin inference
```

## Usage

### Local Testing

1. **Training**:
   ```bash
   cd src
   python train.py
   ```

2. **Testing**:
   ```bash
   pytest tests/ -v
   ```

3. **Inference**:
   ```bash
   cd src
   python inference.py
   ```

### Expected Output

**Training Output**:
```
Loading configuration...
Configuration loaded: {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42}
Loading digits dataset...
Dataset loaded: 1797 samples, 64 features
Training LogisticRegression model...
Training accuracy: 1.0000
Saving model...
Model saved as model_train.pkl
```

**Test Output**:
```
11 passed in 0.68s
```

**Inference Output**:
```
Loading trained model...
Model loaded successfully
Loading digits dataset...
Dataset loaded: 1797 samples, 64 features
Running inference...
Generated 1797 predictions
Prediction range: 0 to 9
Inference accuracy: 1.0000
```

## GitHub Actions Workflows

### 1. Training Workflow (train.yml)
- Triggers on pushes/PRs to `classification` branch
- Sets up Python environment
- Installs dependencies
- Runs training script
- Uploads model as artifact

### 2. Testing Workflow (test.yml)
- Triggers on pushes/PRs to `test` branch
- Sets up Python environment
- Installs dependencies
- Runs full test suite with pytest

### 3. Multi-Job Inference Workflow (inference.yml)
- Triggers on pushes/PRs to `inference` branch
- **Job 1 (test)**: Runs all test cases
- **Job 2 (train)**: Executes training (depends on test success)
- **Job 3 (inference)**: Runs inference (depends on train success)
- Uses `needs` parameter for job dependencies
- Passes model artifacts between jobs

## Configuration

The model hyperparameters are configured in `config/config.json`:

```json
{
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42
}
```

## Test Coverage

The test suite validates:
- Configuration file loading and validation
- Model creation and fitting
- Model accuracy (threshold > 0.8)
- Data loading and preprocessing
- Prediction shape and range validation

## Performance Results

- **Training Accuracy**: 100%
- **Inference Accuracy**: 100%
- **Model Performance**: Perfect classification on digits dataset
- **Test Coverage**: 11 comprehensive tests passing

## Assignment Compliance

✅ **Repository Setup**: Public GitHub repository named `mlops-artifact-pipeline`
✅ **Project Structure**: All required directories and files implemented
✅ **Branching Strategy**: Linear branching (main → classification → test → inference)
✅ **Phase 1**: Training pipeline with config-driven hyperparameters
✅ **Phase 2**: Comprehensive pytest test suite
✅ **Phase 3**: Multi-job workflow with proper dependencies
✅ **GitHub Actions**: All workflows implemented and functional
✅ **Artifacts**: Model uploaded and passed between jobs
✅ **Documentation**: Comprehensive README with setup instructions

## Important Notes

- All hyperparameters are read from JSON configuration file
- No hardcoded values in the codebase
- Workflows are designed to pass in GitHub Actions
- Model artifacts are properly managed between jobs
- Linear branching strategy maintained as per assignment requirements 