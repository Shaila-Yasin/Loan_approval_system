# Loan_approval_system
End-to-end ML pipeline for predicting loan approval.
# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved or rejected based on applicant financial and demographic data.

---

## 📌 Project Overview

This project follows a complete end-to-end ML pipeline including exploratory data analysis, feature engineering, and model training.  
Three classification models were trained and compared to find the best performing one.

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 86.2% |
| Decision Tree | 91.7% |
| Random Forest | 95.9% ✅ |

The **Random Forest** model was selected as the final model due to its highest accuracy and best precision-recall balance across both classes.

---

## 📂 Dataset

The dataset used in this project is publicly available on Kaggle.  
👉 [Download Dataset](<paste your kaggle link here>)


The dataset contains loan application records with the following features:

| Column | Description |
|--------|-------------|
| Age | Applicant age |
| Income | Annual income |
| LoanAmount | Requested loan amount |
| CreditScore | Credit score |
| Gender | Applicant gender |
| Education | Education level |
| EmploymentType | Type of employment |
| City | Applicant city |
| LoanApproved | Target variable (0 = Rejected, 1 = Approved) |

---

## 📁 Project Structure

loan-approval-prediction/
│
├── eda.ipynb                  # Exploratory Data Analysis
├── feature_engineering.ipynb  # Outlier treatment, encoding, scaling
├── model_training.ipynb       # Model training and evaluation
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files excluded from version control
└── README.md                  # Project documentation


---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from the Kaggle link above, place it in the root folder and rename it to `cleaned_loan_data.csv`

### 4. Run notebooks in this order
1. data_cleaning.ipynb
2. eda.ipynb
3. feature_engineering.ipynb   → generates final_dataset.csv and scaler.pkl
4. model_training.ipynb        → generates model.pkl
5. predict.ipynb               → to predict output

> ⚠️ Each notebook depends on the output of the previous one. Run them in order.

---

## 🔬 Methodology

### Exploratory Data Analysis
- Inspected class distribution of target variable `LoanApproved`
- Visualized distributions and outliers for all numerical features
- Analyzed categorical features against loan approval outcome
- Generated correlation heatmap

### Feature Engineering
- **Outlier Treatment** — IQR capping on `Income`, `LoanAmount`, `CreditScore`
- **Encoding** — One-hot encoding on all categorical columns
- **Scaling** — StandardScaler applied to numerical features, scaler saved as `scaler.pkl`

### Model Training
- **SMOTE** applied on training data only to handle class imbalance
- Three models trained and evaluated: Logistic Regression, Decision Tree, Random Forest
- Models evaluated using precision, recall, F1-score and accuracy

---

## 📊 Results

### Logistic Regression
- Accuracy: **86.2%**
- Good baseline but weaker recall on the minority class

### Decision Tree
- Accuracy: **91.7%**
- Better performance but prone to overfitting without pruning

### Random Forest ✅ Best Model
- Accuracy: **95.9%**
- Best precision and recall across both classes
- Saved as `model.pkl` for deployment

---

## 🛠️ Tech Stack

- **Language** — Python 3
- **Libraries** — pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib
- **Environment** — Jupyter Notebook

---

## 👤 Author

**<Shaila_Yasin>**  
[GitHub]() 
