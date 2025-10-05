# ML-Ops Assignment 1: Automated House Price Prediction

This repository implements a complete MLOps workflow for predicting house prices using the Boston Housing dataset and classic scikit-learn regression models. The project demonstrates modularity, version control best practices (branching and merging), and continuous integration via GitHub Actions.

## ⚙️ Setup and Installation

Follow these steps to set up your local environment and install required dependencies.

1. Repository setup
```bash
# Clone the repository
git clone https://github.com/Krishanu-Git/ML-Ops-Assignment-1.git
cd ML-Ops-Assignment-1
```

2. Environment activation
```bash
# Create and activate a dedicated Conda environment with Python 3.9
conda create -n mlops_a1 python=3.9 -y
conda activate mlops_a1
```

3. Install dependencies
All required packages (scikit-learn, pandas, numpy) are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

## ▶️ How to Run

The core pipeline logic is implemented as generic functions in `misc.py` (data loading, preprocessing, training, testing).

1. Training individual models
Run model-specific scripts to train and print performance.

- Decision Tree Regressor
```bash
python train.py
# Branch origin: dtree (merged to main)
```

- Kernel Ridge
```bash
python train2.py
# Branch origin: kernelridge
```

2. Generate a performance comparison report
`compare_models.py` executes both training scripts and prints a consolidated comparison (Mean Squared Error). This is the primary script used by CI.
```bash
python compare_models.py
```

## 🚀 MLOps Automation (GitHub Actions CI)

A CI workflow is configured to validate code and model performance.

- Workflow file: `.github/workflows/ci.yml`
- Trigger: any push event to the `kernelridge` branch
- Pipeline steps:
    1. Checkout code
    2. Install dependencies from `requirements.txt`
    3. Run `compare_models.py` to train both models and display the performance comparison report

## Project layout (key files)
- README.md
- misc.py               — shared pipeline utilities (load, preprocess, train, test)
- train.py              — trains DecisionTreeRegressor
- train2.py             — trains KernelRidge
- compare_models.py     — runs both training scripts and compares MSE
- requirements.txt
- .github/workflows/ci.yml

## Notes
- The dataset used is the Boston Housing dataset via scikit-learn.
- Keep model experiments and CI changes isolated to feature branches (e.g., `dtree`, `kernelridge`) and merge to `main` only after review/testing.
- `compare_models.py` is used by CI to ensure reproducible end-to-end execution and to monitor model performance regressions.
