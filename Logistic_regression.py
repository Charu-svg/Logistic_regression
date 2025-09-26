# Breast Cancer Classification using Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Class distribution:")
print(y.value_counts(normalize=True))

# -----------------------------
# 2. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. Train Logistic Regression Model
# -----------------------------
model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------
y_probs = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1
y_pred_default = (y_probs >= 0.5).astype(int)       # default threshold = 0.5

# -----------------------------
# 6. Evaluation Metrics
# -----------------------------
cm = confusion_matrix(y_test, y_pred_default)
report = classification_report(y_test, y_pred_default, target_names=data.target_names)
precision = precision_score(y_test, y_pred_default)
recall = recall_score(y_test, y_pred_default)
f1 = f1_score(y_test, y_pred_default)
roc_auc = roc_auc_score(y_test, y_probs)

print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# -----------------------------
# 7. Feature Importance
# -----------------------------
coef_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance:\n", coef_df)

# -----------------------------
# 8. Confusion Matrix Plot
# -----------------------------
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = data.target_names
plt.xticks([0,1], classes)
plt.yticks([0,1], classes)
plt.xlabel('Predicted')
plt.ylabel('True')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.show()

# -----------------------------
# 9. ROC Curve Plot
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
plt.plot([0,1], [0,1], linestyle='--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Save Model & Scaler
# -----------------------------
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved successfully!")

# -----------------------------
# 11. Reload Model & Scaler
# -----------------------------
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
print("Model and scaler loaded successfully!")

# -----------------------------
# 12. Predict New Data
# -----------------------------
# Example: new sample (replace with actual values)
new_data = [[14.0, 20.0, 90.0, 600.0, 0.1, 0.2, 0.3, 0.15, 0.2, 0.07,
             0.3, 1.2, 2.0, 25.0, 0.005, 0.03, 0.03, 0.02, 0.02, 0.004,
             16.0, 25.0, 100.0, 700.0, 0.]]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
prediction_prob = model.predict_proba(new_data_scaled)[:, 1]

print("\nPredicted class:", prediction[0])
print("Predicted probability of class 1 (benign):", prediction_prob[0])

# -----------------------------
# 13. Optional: Cross-Validation
# -----------------------------
scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print("\n5-Fold CV ROC-AUC scores:", scores)
print("Mean CV ROC-AUC:", scores.mean())
