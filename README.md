# Breast Cancer Classification using Logistic Regression

## Project Objective
Build a binary classifier to predict whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** using the **Breast Cancer Wisconsin dataset** and **Logistic Regression**.

---

## Dataset Description
- **Source:** Scikit-learn built-in dataset (`load_breast_cancer`)  
- **Samples:** 569  
- **Features:** 30 numerical features (mean, standard error, and worst value of cell nuclei measurements)  
- **Target:** Binary (`0 = malignant`, `1 = benign`)  

Example features:  
`mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`  

---

## Project Steps

### 1. Load Dataset
- Imported dataset using `sklearn.datasets.load_breast_cancer`
- Converted into Pandas DataFrame for easy inspection

### 2. Train/Test Split
- Split data into **80% training** and **20% testing**  
- Stratified split to maintain class balance

### 3. Feature Scaling
- Standardized features using `StandardScaler`  
- Ensures all features contribute equally to the model

### 4. Logistic Regression Model
- Trained Logistic Regression (`solver='liblinear'`) on scaled training data  
- Handled class imbalance using `class_weight='balanced'`  

### 5. Predictions
- Predicted probabilities and binary outcomes  
- Default threshold = 0.5

### 6. Evaluation Metrics
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **ROC Curve & AUC**
- **Threshold Tuning**: Adjusted threshold to maximize F1-score  

Example:
