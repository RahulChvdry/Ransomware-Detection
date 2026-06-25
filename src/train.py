import pandas as pd
import numpy as np

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# =========================
# 1. LOAD + CLEAN DATA
# =========================
df = pd.read_csv("../data/22_Ransomware_Detection_Using_Features_of_PE_Imports_2.csv", header=0)

df = df.drop(columns=[df.columns[0]])

if "SHA256" in df.columns:
    df = df.drop(columns=["SHA256"])

df['label'] = df['label'].map({'M': 1, 'B': 0})
df = df.fillna(0)

# =========================
# 2. SPLIT DATA
# =========================
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 3. HANDLE IMBALANCE (SMOTE)
# =========================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:\n", y_train.value_counts())

# =========================
# 4. SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 5. TRAIN MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True, class_weight='balanced')
}

# =========================
# 6. EVALUATION
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))