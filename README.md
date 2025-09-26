#  Breast_cancer Classification using Logistic Regression

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


---

## 📌 How to Run  

### 1. Clone or Download Project  
```bash
git clone https://github.com/your-username/Breast_Cancer_Logistic_Regression.git
cd Breast_Cancer_Logistic_Regression
### 2.Create Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
### 3.Install Dependencies
pip install -r requirements.txt
### 4.Run the Script
python Logistic_regression.py
### 5. Test with New Data
new_data = [[14.0, 20.0, 90.0, 600.0, ..., 0.08]]  # 30 values


