# Customer Churn Prediction

## 📝 Project Overview
This project predicts whether a customer will **churn** (leave) a subscription-based service using historical data.  
The goal is to help businesses **retain customers** and **reduce revenue loss** by identifying high-risk customers before they leave.

**Dataset used:** [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

---

## 📊 Features in Dataset

| Feature Name         | Description |
|---------------------|-------------|
| CreditScore         | Customer credit score |
| Geography           | Country (France, Spain, Germany) |
| Gender              | Male/Female |
| Age                 | Age of customer |
| Tenure              | Number of years with the bank |
| Balance             | Account balance |
| NumOfProducts       | Number of products subscribed |
| HasCrCard           | Customer has a credit card (1/0) |
| IsActiveMember      | Customer is active (1/0) |
| EstimatedSalary     | Estimated annual salary |
| Exited              | Target variable: 1 = churn, 0 = stayed |

---

## 🔧 Steps Followed

1. **Data Loading**: Load CSV dataset using `pandas`.  
2. **Data Cleaning**: Removed unnecessary columns: `RowNumber`, `CustomerId`, `Surname`.  
3. **Encoding Categorical Features**: Converted `Gender` and `Geography` into numeric variables using One-Hot Encoding.  
4. **Train-Test Split**: Split data into **80% training** and **20% testing** sets.  
5. **Model Training**: Trained three models:
   - Logistic Regression  
   - Random Forest Classifier  
   - Gradient Boosting Classifier  
6. **Model Evaluation**: Evaluated using **Accuracy, Confusion Matrix, Precision, Recall, F1-score**. Focused on **Recall for churners**.  
7. **Optional Improvements**: Hyperparameter tuning, threshold adjustment, feature engineering.

---

## ⚙️ How to Run

1. Clone/download the project folder.  
2. Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
