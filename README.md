# **📱 Customer Churn Prediction – Telecommunications Industry**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-006699?logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-2E8B57)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)

Predicting customer churn for a telecom company using machine learning.

**Goal:** Identify customers likely to leave and provide actionable insights to reduce churn and improve retention.

# **🎯 Business Problem**

Customer churn is a major challenge for telecom companies, leading to lost revenue and higher acquisition costs.
By predicting which customers are likely to churn, the company can take proactive measures such as targeted offers, service improvements, or personalized engagement.

# **📌 Project Objectives**

1. Predict which customers are likely to churn using machine learning models.

2. Identify key features that influence churn to inform business decisions.

3. Build a deployable model that can be used to proactively retain customers.

4. Provide actionable insights and recommendations to reduce churn and increase customer retention.

# **🧠 Machine Learning Workflow**

- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training with multiple algorithms  
- Evaluation and comparison  
- Selection of the best performing model 

# **🗂️ Dataset Overview**

**Source:** SyriaTel customer data

### **Preprocessing Steps:**

1. Handled missing values and inconsistent entries

2. Encoded categorical variables

3. Created additional features like total charges, total minutes, and tenure

# **📊 Exploratory Data Analysis (EDA)**

## **Key Visuals**

### **Churn Distribution**

<img width="627" height="545" alt="image" src="https://github.com/user-attachments/assets/6601e2a1-8726-4e96-a084-fbec2b7a3e56" />

### **Minutes vs Charges**

<img width="687" height="545" alt="image" src="https://github.com/user-attachments/assets/54f7bc6e-929a-4d41-acb6-ba9a1daf5a87" />

### **Heatmap Correlations**

<img width="625" height="526" alt="image" src="https://github.com/user-attachments/assets/d2b59094-293a-481e-a116-3b070465dff3" />

#### **Key Observations**

- Class Imbalance (Moderate):  While not as extreme as the "Customer Service Category," there's still a noticeable class imbalance.  The "False" (no churn) category has a substantially higher count than the "True" (churn) category.

- Strong Positive Correlation: The scatter plot clearly shows a strong positive correlation between "Total Minutes" and "Total Charges." As "Total Minutes" increases, "Total Charges" also tends to increase. This suggests that customers who use more minutes tend to have higher total charges.

-  Strong Positive Correlation Between Total Minutes and Total Charges: The most prominent feature is the strong positive correlation (0.89) between Total_Minutes and Total_Charges. 

# **🧩 Modeling Approach**

**Models Tested:** Logistic Regression, Decision Tree, Random Forest, XGBoost

**Preprocessing:** Feature scaling, encoding, handling class imbalance and fine-tuning

**Selection:** Random Forest chosen based on best accuracy, recall, and F1-score

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 78.7%    | 40.0%     | 81.0%  | 53.0%    |
| Decision Tree      | 94.6%    |  88.5%   | 91.5%  | 90.0%    |
| Random Forest      | 97.9%    | 99.0%     | 93.0%  | 96.0%    |
| XGBoost            | 98.0%    | 99.0%    |  94.0%  | 96.0%   |

Ensemble models (Random Forest and XGBoost) clearly outperform simpler models. XGBoost edges out slightly in accuracy, but both Random Forest and XGBoost provide highly reliable predictions, making them ideal choices for deployment and actionable churn prevention strategies.

## **Confusion Matrix**

<img width="539" height="453" alt="image" src="https://github.com/user-attachments/assets/bfdc633c-cb6f-466b-b492-7323f6c90556" />

**Insight**: The model is particularly effective at identifying customers who will not churn (100% recall for "Not Churn" class) and exhibits perfect precision when predicting churn (100%), meaning any customer flagged as likely to churn is a true positive case.

## **ROC-AUC Curve**

<img width="691" height="545" alt="image" src="https://github.com/user-attachments/assets/2ced4c62-65de-4403-8d63-6e38d9c5953b" />

 **Insight**: Random Forest model offers slightly better performance over the base model, as indicated by a marginally higher Area Under the Curve (AUC) value of 0.94 compared to 0.93.

## **Feature Importance graph**

<img width="1020" height="545" alt="image" src="https://github.com/user-attachments/assets/80b406fd-c4c5-4c41-8d4e-aebeace68a13" />

**Dominant Feature:** Total_Charges is overwhelmingly the most important feature, dwarfing all others. This indicates it's the strongest predictor in the model.

# **💡 Key Insights / Business Recommendations**

1. High total charges increase churn probability: Customers with higher bills are more likely to leave — consider offering personalized discounts or flexible payment plans.

2. Customer service calls are a strong predictor: Customers who frequently contact support are at higher risk of churn — proactively address their concerns.

3. Multiple services reduce churn risk: Customers subscribed to multiple services are less likely to leave — consider bundling services to improve retention.

4. Dependence on contract type: Customers on month-to-month plans tend to churn more than those on long-term contracts — encourage contract upgrades with incentives.

5. Targeted retention strategies: Focus on customers with high total charges and frequent service calls to reduce churn effectively.

# **🚀 Deployment / App**

**Run Locally:**

    git clone https://github.com/kennethnyangweso/Project-Predicting-customer-churn-in-the-Telecommunications-Industry.git
    cd Project-Predicting-customer-churn-in-the-Telecommunications-Industry
    pip install -r requirements.txt
    python app.py

**App Overview:**

Users can input customer details (e.g., services subscribed, charges, contract type) and get a predicted churn probability.

### **Expected Results**

<img width="1366" height="649" alt="Screenshot (41)" src="https://github.com/user-attachments/assets/1fea6707-3f17-4255-abee-a83f25790255" />


# **📁 Project Structure**

    ├── data/                   # Raw and cleaned datasets
    ├── notebooks/              # EDA and modeling notebooks
    ├── models/                 # Saved model(s)
    ├── app.py                  # Deployment app
    ├── Visualizations/         # Key plots for README & notebook
    ├── README.md
    ├── requirements.txt        # Python dependencies

# **🛠️ Tools & Technologies**

- Python: Pandas, NumPy, Scikit-learn

- Visualization: Matplotlib, Seaborn

- Modeling: Random Forest, Decision Tree, XGBoost, Logistic Regression

- Deployment: Streamlit (Python web app)

- IDE: Jupyter Notebook

## **👤 Author**

**Kenneth Nyangweso**

**Data Scientist | Electrical & Telecommunications Engineer**
