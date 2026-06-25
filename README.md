
# Ransomware Detection using Static PE Analysis

A machine learning-based static analysis framework to detect ransomware 
from Windows Portable Executable (PE) files without executing them.

## Overview
Traditional signature-based detection struggles against new or modified 
ransomware variants. This project uses static PE features — imports, 
string artifacts, and resource characteristics — to classify files as 
ransomware or benign using machine learning classifiers.

## Features
- Static analysis using 3 PE feature groups: imports, strings, resources
- Trained and evaluated 4 ML classifiers: Random Forest, Logistic 
  Regression, Decision Tree, SVM
- TF-IDF vectorization applied on PE string data
- SMOTE used to handle class imbalance
- Best models saved using joblib for deployment

## Results
| Dataset | Best Model | Accuracy | AUC |
|---------|-----------|----------|-----|
| PE Imports | Random Forest | 100% | 0.993 |
| PE Strings | Random Forest + TF-IDF | 96% | 0.995 |
| PE Resources | Random Forest | - | - |

## Tech Stack
- Python, scikit-learn, pandas, NumPy
- Matplotlib, Seaborn (visualization)
- imbalanced-learn (SMOTE)
- joblib (model saving)

## Project Structure
- `data/` — Datasets (PE imports, strings, resources)
- `models/` — Trained model files (.pkl)
- `outputs/` — Result graphs and screenshots
- `src/eda.py` — Exploratory data analysis
- `src/train.py` — Training on PE imports dataset
- `src/train_any_dataset.py` — Generic training script
- `src/train_resources.py` — Training on PE resources dataset
- `ResearchPaper.pdf` — Reference research paper

## How to Run
1. Install dependencies:
   
   pip install pandas scikit-learn imbalanced-learn matplotlib seaborn joblib
   
2. Navigate to `src/` folder
3. Run EDA: `python eda.py`
4. Run training: `python train.py`

## Team
Group project — IIITDM Jabalpur | Supervised by Dr. Nitish Andola



