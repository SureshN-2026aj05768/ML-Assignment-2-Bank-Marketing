# Machine Learning Assignment 2
## Bank Term Deposit Subscription Prediction

This project is developed as part of **Machine Learning Assignment 2**.

The objective of this project is to develop, evaluate, and compare multiple machine learning classification models for predicting whether a bank client will subscribe to a term deposit.

The project also includes a **Streamlit web application** that loads the saved trained models and evaluates them on uploaded test data without retraining.

---

## 1. Problem Statement

The objective is to predict whether a bank client will subscribe to a term deposit based on client information and marketing campaign attributes.

This is treated as a **binary classification problem**:

- `yes` - Client subscribed to the term deposit
- `no` - Client did not subscribe to the term deposit

Five classification algorithms are implemented and compared using the same training and testing data.

---

## 2. Dataset Description

The project uses the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

### Dataset Summary

| Item | Details |
|---|---|
| Dataset | Bank Marketing |
| Source | UCI Machine Learning Repository |
| Dataset File | `bank-full.csv` |
| Total Records | 45,211 |
| Input Features | 16 |
| Target Variable | `y` |
| Classification Type | Binary Classification |
| Missing Values | None |
| Duplicate Records | None detected in the dataset used |

The target variable indicates whether the client subscribed to a term deposit.

### Target Distribution

The dataset is imbalanced, with approximately:

- **88.3%** - `no`
- **11.7%** - `yes`

Therefore, model performance is evaluated using multiple metrics rather than Accuracy alone.

---

## 3. Data Preprocessing

The following preprocessing steps are applied:

1. The target variable `y` is encoded:
   - `no = 0`
   - `yes = 1`

2. The dataset is divided into:
   - **80% Training Data**
   - **20% Testing Data**

3. Stratified sampling is used during the train-test split to maintain the target-class distribution.

4. Numerical features are standardized using:

   `StandardScaler`

5. Categorical features are encoded using:

   `OneHotEncoder(handle_unknown="ignore")`

6. Numerical and categorical transformations are combined using:

   `ColumnTransformer`

7. The preprocessing stage and each classifier are combined using a Scikit-learn `Pipeline`.

This ensures that the same preprocessing applied during model training is also used during deployment.

---

## 4. Classification Models

The following five classification models are implemented:

1. **Logistic Regression**
2. **Decision Tree**
3. **K-Nearest Neighbors (KNN)**
4. **Gaussian Naive Bayes**
5. **Support Vector Machine (SVM)**

All models are trained and evaluated using the same training and testing split.

---

## 5. Evaluation Metrics

Each model is evaluated using the following metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Confusion Matrix and Classification Report are also generated for detailed model evaluation.

---

## 6. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | **0.9056** | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | **0.4701** | 0.3990 |
| K-Nearest Neighbors | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Gaussian Naive Bayes | 0.8548 | 0.8101 | 0.4059 | **0.5198** | 0.4559 | 0.3774 |
| Support Vector Machine | **0.9049** | 0.9037 | **0.6875** | 0.3431 | 0.4578 | **0.4420** |

---

## 7. Model Observations

### Logistic Regression

Logistic Regression achieved an Accuracy of **0.9012** and the highest AUC of **0.9056**. It showed strong overall classification performance, although Recall for the positive subscription class was comparatively low.

### Decision Tree

Decision Tree achieved an Accuracy of **0.8746** and the highest F1 Score of **0.4701**. Its Recall was higher than Logistic Regression, KNN, and SVM, although its AUC was the lowest among the five models.

### K-Nearest Neighbors

KNN achieved an Accuracy of **0.8962** and an AUC of **0.8277**. The overall performance was good, but Recall for the positive class remained relatively low.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved the highest Recall of **0.5198**, indicating that it identified a larger proportion of positive subscription cases. However, its Accuracy and Precision were lower than the other leading models.

### Support Vector Machine

Support Vector Machine achieved the highest Accuracy of **0.9049**, highest Precision of **0.6875**, and highest MCC of **0.4420**. Its AUC of **0.9037** was also very close to the highest AUC obtained by Logistic Regression.

### Overall Observation

Support Vector Machine provided the strongest overall performance based on Accuracy, Precision, AUC, and MCC.

However, there is no single model that leads on every evaluation metric:

- **Highest Accuracy:** Support Vector Machine
- **Highest AUC:** Logistic Regression
- **Highest Precision:** Support Vector Machine
- **Highest Recall:** Gaussian Naive Bayes
- **Highest F1 Score:** Decision Tree
- **Highest MCC:** Support Vector Machine

Since the target classes are imbalanced, the models are evaluated using multiple metrics rather than Accuracy alone.

---

## 8. Model Deployment

After training, the complete Scikit-learn pipelines are saved using `joblib`.

The following trained model files are included:

```text
model/
├── logistic_regression.joblib
├── decision_tree.joblib
├── knn.joblib
├── naive_bayes.joblib
└── svm.joblib
```

Each saved object contains the preprocessing steps and the trained classifier.

The saved model pipeline is loaded directly by the Streamlit application.

**The uploaded test data is used only for prediction and evaluation. The model is not retrained on the uploaded test data.**

---

## 9. Test Dataset

A held-out test dataset is provided as:

```text
test_data.csv
```

The test dataset contains:

- **9,043 records**
- **16 input features**
- Target column `y`

The file can also be downloaded directly from the Streamlit application and used to test any of the five saved classification models.

---

## 10. Streamlit Application

A Streamlit web application is developed to provide interactive model evaluation.

### Application Features

The application allows the user to:

- Select one of the five trained classification models
- Upload a CSV test dataset
- Download the provided sample test dataset
- View the expected CSV format and required columns
- View the uploaded test-data summary
- View information about the selected classification model
- Evaluate the selected saved model
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- View a Confusion Matrix containing counts and class-wise percentages
- View the Classification Report
- Perform evaluation without retraining the selected model

### Live Streamlit Application

[Open Bank Term Deposit Subscription Prediction App](https://sureshn-2026aj05768-ml-assignment-2-bank-marketing.streamlit.app/)

---

## 11. Expected Test Data Format

The uploaded CSV should contain the following 16 input features:

### Numerical Features

```text
age
balance
day
duration
campaign
pdays
previous
```

### Categorical Features

```text
job
marital
education
default
housing
loan
contact
month
poutcome
```

### Target Variable

```text
y
```

Expected target values:

```text
yes
no
```

The provided `test_data.csv` can be used as a reference for the expected input structure.

---

## 12. Repository Structure

```text
ML-Assignment-2-Bank-Marketing/
│
├── data/
│   └── bank-full.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
├── ML_Assignment2.ipynb
├── app.py
├── requirements.txt
├── test_data.csv
├── README.md
└── .gitignore
```

---

## 13. Running the Application

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app.py
```

The application will start locally and can then be accessed using the URL displayed by Streamlit.

---

## 14. Technology Stack

The project uses:

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook
- GitHub

---

## 15. Deployment Environment

The machine learning workflow was developed and tested through the notebook environment.

The GitHub repository and deployed Streamlit application were also accessed and validated from the **BITS Pilani Prayogshala Virtual Machine** as part of the assignment execution and deployment verification.

---

## 16. Project Workflow

```text
UCI Bank Marketing Dataset
           ↓
Dataset Exploration
           ↓
Data Preprocessing
           ↓
Train / Test Split
           ↓
Five Classification Models
           ↓
Model Evaluation
           ↓
Six-Metric Model Comparison
           ↓
Save Trained Pipelines
           ↓
Generate Held-Out Test Dataset
           ↓
Streamlit Application
           ↓
Upload Test CSV
           ↓
Load Selected Saved Model
           ↓
Prediction and Evaluation
           ↓
Metrics + Confusion Matrix + Classification Report
```

---

## 17. Dataset Reference

**UCI Machine Learning Repository**  
Bank Marketing Dataset

https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## 18. Conclusion

Five machine learning classification models were successfully developed and evaluated using the UCI Bank Marketing dataset.

The evaluation demonstrates that model performance varies depending on the selected metric. Support Vector Machine produced the strongest overall combination of Accuracy, Precision, AUC, and MCC, while other models performed better for specific measures such as Recall and F1 Score.

The trained pipelines were successfully saved and integrated into a Streamlit application that performs predictions and evaluation on uploaded test data without retraining the models.
