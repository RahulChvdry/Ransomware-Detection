import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE


# =========================
# 🔥 MAIN FUNCTION
# =========================
def train_model(file_path, target_column):
    
    # 1. LOAD DATA
    df = pd.read_csv(file_path)
    print("\nDataset Shape:", df.shape)

    # 2. BASIC CLEANING
    df = df.dropna()

    # Convert categorical target if needed
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].astype('category').cat.codes

    # 3. SPLIT
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. HANDLE IMBALANCE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # 5. SCALING
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 6. MODELS
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True)
    }

    # 7. TRAIN + EVALUATE
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(f"\n{name}")
        print(classification_report(y_test, y_pred))
        print("AUC:", roc_auc_score(y_test, y_prob))


# =========================
# 🔥 RUN HERE
# =========================

train_model("../data/65_Ransomware_Detection_Using_PE_Resources.csv", "label")