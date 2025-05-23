# -*- coding: utf-8 -*-
"""Predictive Analysis_Faiza Adinda.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TgBEt8Ah3tYnnEuEQ8qQDFtf9OMMgTq_

# Import Libary
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, precision_score, recall_score, f1_score

"""# Data Loading"""

# load the dataset
heart = pd.read_csv('heart_disease.csv')
heart

"""# Data Understanding"""

heart.info()

heart.describe()

# Cek apakah ada nilai duplikat
heart.duplicated().sum()

# Memilih fitur-fitur numerik yang rawan outlier
features_to_check = ['resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']

# Atur ukuran visualisasi
plt.figure(figsize=(15, 10))

# Buat boxplot untuk tiap fitur
for i, feature in enumerate(features_to_check, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=heart, y=feature, color='skyblue')
    plt.title(f'Boxplot of {feature}')
    plt.tight_layout()

plt.show()

# Univariate Analysis
df_numeric = heart.select_dtypes(include=['int64', 'float64'])
df_numeric.hist(figsize=(15, 12), bins=20, edgecolor='black')
plt.suptitle("Histogram Setiap Fitur", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# Multivariate Analysis
sns.pairplot(df_numeric)
plt.suptitle("Pairplot antar Fitur", y=1.02)
plt.show()

# Heatmap Korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Heatmap Korelasi Antar Fitur Numerik")
plt.show()

"""# Data Preparation"""

# Menghapus nilai duplikat
heart = heart.drop_duplicates()
heart.duplicated().sum()

# Menangani Outlier
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Kolom yang ingin dibersihkan dari outlier
outlier_cols = ['resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
heart = remove_outliers_iqr(heart, outlier_cols)

heart.describe()

# Split Dataset dan Cek Jumlah Sampel
X = heart.drop('target', axis=1)
y = heart['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

# Melakukan Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# Modelling"""

# Membangun Model Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Membangun Model Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Membangun Model Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

"""# Evaluasi"""

# Logistic Regression
print("Logistic Regression:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_logreg), 2))
print("Precision:", round(precision_score(y_test, y_pred_logreg), 2))
print("Recall:", round(recall_score(y_test, y_pred_logreg), 2))
print("F1 Score:", round(f1_score(y_test, y_pred_logreg), 2))
print()

# SVM
print("SVM:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_svm), 2))
print("Precision:", round(precision_score(y_test, y_pred_svm), 2))
print("Recall:", round(recall_score(y_test, y_pred_svm), 2))
print("F1 Score:", round(f1_score(y_test, y_pred_svm), 2))
print()

# Random Forest
print("Random Forest:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf), 2))
print("Precision:", round(precision_score(y_test, y_pred_rf), 2))
print("Recall:", round(recall_score(y_test, y_pred_rf), 2))
print("F1 Score:", round(f1_score(y_test, y_pred_rf), 2))
print()