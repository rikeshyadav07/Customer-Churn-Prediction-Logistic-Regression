# Customer Churn Prediction: Telecom Industry Analysis

## Project Overview

This repository hosts a machine learning project focused on predicting customer attrition (churn) within a telecommunications company. The primary objective is to develop a robust classification model that identifies customers at high risk of leaving, enabling the company's marketing and retention teams to execute targeted intervention strategies.

The analysis is performed using a **Logistic Regression** model, providing not only accurate predictions but also transparent insights into the underlying drivers of customer churn.

### Project Status

| Metric | Status |
| :--- | :--- |
| **Model Type** | Binary Classification |
| **Primary Model** | Logistic Regression |
| **Accuracy** | 79.67% |
| **ROC AUC Score** | 83.28% |

---

## Goal and Problem Statement

**Goal:** To build a predictive model capable of classifying customers as Churn (`Yes`) or Non-Churn (`No`).

**Problem:** Customer churn is a major financial drain for telecom providers. Identifying potential churners proactively is crucial for maximizing Customer Lifetime Value (CLV) and minimizing revenue loss associated with service cancellations.

---

## Dataset

The analysis uses the publicly available **`TelecomCustomerChurn.csv`** dataset, which contains records for 7,043 customers and includes 20 predictor variables.

**Key Features include:**
* **Demographic Info:** Gender, SeniorCitizen, Partner, Dependents.
* **Services Used:** PhoneService, MultipleLines, InternetService (DSL, Fiber optic, No), OnlineSecurity, etc.
* **Account Info:** Tenure, Contract (Month-to-month, One year, Two year), PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.
* **Target Variable:** **Churn** (Yes/No).

---

## Methodology and Implementation

The project follows a standard machine learning pipeline:

1.  **Data Loading and Initial Exploration:** Loaded data into a Pandas DataFrame and inspected data types and missing values.
2.  **Data Preprocessing:**
    * Handled missing values in the `TotalCharges` column by dropping the small number of associated rows.
    * Converted the `TotalCharges` column to a numeric type.
3.  **Feature Engineering:**
    * Converted all categorical features (e.g., `Contract`, `InternetService`, `PaymentMethod`) into a numerical format using **One-Hot Encoding** for model compatibility.
4.  **Data Splitting:** The data was split into training and testing sets (70/30 ratio) using stratified sampling to ensure the proportion of churners is maintained in both sets.
5.  **Model Training:** A **Logistic Regression** model was initialized and trained on the scaled training data.
6.  **Evaluation:** Model performance was assessed using a confusion matrix and several key classification metrics.

---

## Key Results and Insights

### Model Performance Metrics

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 79.67% | Overall proportion of correct predictions. |
| **ROC AUC** | 83.28% | Measures the model's ability to distinguish between churners and non-churners. |
| **Precision** | 65.28% | Out of all predicted churners, 65.28% were correct. |
| **Recall** | **50.27%** | **Critical Challenge:** The model only correctly identified about half of the actual churners. |
| **F1-Score** | 56.80% | Harmonic mean of Precision and Recall. |

### Feature Importance & Churn Drivers

Analysis of the Logistic Regression coefficients revealed the strongest predictors:

| Driver Type | Key Features | Impact on Churn |
| :--- | :--- | :--- |
| **High Risk** | `InternetService_Fiber optic` | Strongest positive predictor. Fiber optic customers are significantly more likely to churn. |
| | `MonthlyCharges` (High) | Higher monthly costs drive churn. |
| | `Tenure` (Low) | Newer customers are more likely to leave. |
| **Retention Factor** | `Contract_Two year` | Strongest negative predictor. Customers with long-term contracts are highly retained. |
| | `Tenure` (High) | Long-term customers show high loyalty. |

---

## Future Work and Improvements

1.  **Address Low Recall:** Focus on improving the model's ability to identify true churners. This could involve:
    * **Class Imbalance Techniques:** Implementing SMOTE or using class weights during model training.
    * **Algorithm Exploration:** Testing more complex algorithms like **Random Forest** or **Gradient Boosting (XGBoost)**.
2.  **Hyperparameter Tuning:** Systematically optimize the Logistic Regression model parameters using techniques like GridSearchCV.
3.  **Deep Dive on Fiber Optic:** Further investigation into the quality of the Fiber Optic service, as it is the most significant churn driver.

---

## Installation and Usage

### Prerequisites

You will need Python 3.x and the following libraries installed:

```bash
pip install pandas numpy scikit-learn
