import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../data/22_Ransomware_Detection_Using_Features_of_PE_Imports_2.csv", header=0)

print("\nInitial Shape:", df.shape)
print("\nColumns:", df.columns)

# =========================
# 2. CLEAN DATA
# =========================

# Drop ID column
df = df.drop(columns=[df.columns[0]])

# Drop SHA256 column
if "SHA256" in df.columns:
    df = df.drop(columns=["SHA256"])

# Convert label
df['label'] = df['label'].map({'M': 1, 'B': 0})

# Handle missing values
df = df.fillna(0)

print("\nAfter Cleaning Shape:", df.shape)
print(df.head())

# =========================
# 3. BASIC INFO
# =========================
print("\nData Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# =========================
# 4. CLASS DISTRIBUTION
# =========================
print("\nClass Distribution:")
print(df['label'].value_counts())

sns.countplot(x='label', data=df)
plt.title("Class Distribution (0=Benign, 1=Malware)")
plt.show()

# =========================
# 5. CORRELATION HEATMAP
# =========================
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# =========================
# 6. FEATURE DISTRIBUTION (Few Columns)
# =========================
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()