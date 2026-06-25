import pandas as pd

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer

# Save model
import joblib

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../data/60_Ransomware_Detection_Using_Strings_versioned.csv")   # 🔥 change filename

print("Initial Shape:", df.shape)
print(df.head())

# =========================
# 2. CLEAN DATA
# =========================
df = df.drop(columns=["Unnamed: 0", "SHA256"], errors='ignore')

# label already numeric (0/1)
df['label'] = df['label'].astype(int)

# =========================
# 3. TEXT → NUMERIC (TF-IDF)
# =========================
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['strings']).toarray()
y = df['label']

print("\nData transformed using TF-IDF")

# =========================
# 4. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. HANDLE IMBALANCE (SMOTE)
# =========================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# =========================
# 6. TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# =========================
# 7. EVALUATE
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nResults:")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

# =========================
# 8. SAVE MODEL
# =========================
joblib.dump(model, "../models/strings_model.pkl")
joblib.dump(tfidf, "../models/tfidf.pkl")

print("\nModel & TF-IDF saved in models folder ✅")