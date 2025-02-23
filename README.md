**PREDICTING CUSTOMER CHURN IN SYRIATEL TELECOMMUNICATIONS COMPANY:**

- A Machine Learning Approach for Proactive Customer Retention

**BUSINESS UNDERSTANDING:**

**Project Overview:**

Customer retention is a critical concern for telecommunications companies, as acquiring new customers is often more expensive than retaining existing ones. With increasing competition, understanding why customers leave (churn) and identifying those at risk can significantly improve business strategies. This project aims to build a binary classification model to predict customer churn based on various usage patterns and customer service interactions.

**Problem Statement:**

SyriaTel, like many telecommunication providers, faces challenges in retaining customers due to factors such service dissatisfaction, competitive offers,shifting customer needs, and many other factors. Identifying customers who are likely to churn before they actually leave allows the company to take proactive measures such as personalized offers, improved service quality, and better customer engagement. Without a predictive approach, customer retention efforts may be reactive and less effective, leading to revenue losses.

**OBJECTIVES:**

**Main Objective:**

* To develop machine learning models that accurately predicts customer churn in order to help SyriaTel reduce churn rates and improve customer satisfaction.
Specific Objectives:

1. Data Exploration & Cleaning – Understand the dataset structure, handle missing values, and preprocess data for analysis.

2. Feature Engineering & Selection – Identify key factors contributing to churn and optimize input variables for modeling.

3. Model Development – Train and evaluate multiple classification models to determine the best-performing one.
Interpretability & Insights – Analyze the key drivers of churn and provide business recommendations.

  **Why Now?**

- **Increased Competition:** Telecommunications companies are experiencing higher competition, making customer retention more crucial than ever.

- **Data Availability:** SyriaTel has rich historical customer data that can be leveraged for predictive modeling.

- **Cost Efficiency:** Predicting churn allows for targeted interventions, reducing marketing costs while improving customer loyalty.

- Advancements in AI & Machine Learning: Modern machine learning techniques make churn prediction more accurate and actionable than traditional rule-based approaches.

 **Metrics of Success:**

**Accuracy & Precision:** Ensuring the model correctly classifies customers into churn and non-churn categories.

**Recall (Sensitivity):** High recall ensures that most potential churners are identified.

**F1-score:** Balancing precision and recall to optimize performance.

**ROC-AUC Score:** Measuring the model’s ability to distinguish between churn and non-churn customers.

**Business Impact:** Reduction in churn rates and improved customer engagement based on model-driven interventions.

**DATA UNDERSTANDING:**

The SyriaTel Customer Churn Dataset contains information about customers, their usage patterns, service subscriptions, and interactions with customer service. The dataset is structured with 3,333 records and 21 features, including the target variable (Churn), which indicates whether a customer has left the service or not.

**DATA PREPARATION:**

This phase involves transforming raw data into a structured and clean format suitable for modeling. This step ensures that the dataset is free from inconsistencies, missing values, and unnecessary variables while preparing it for machine learning algorithms.

**Python Libraries used:**

- Pandas - Used for data manipulation and analysis

- Numpy- Used for multi-dimensional arrays e.g. matrices

- Matplotlib-Used for interactive visualizations

- Seaborn-Used for advanced visualizations

- Scikit-learn- Used for preprocessing, modeling and evaluation



**Data Cleaning**

This is the process of cleaning the dataset by:

- Dropping unnecessary columns

- Checking and dealing with missing values

- Dropping the duplicates

- Changing the columns format

- Checking for outliers

**Feature Engineering:**

Feature engineering is crucial for improving model accuracy and interpretability. In this project, the goal was to extract meaningful insights from the available features and create new ones that enhance the model's predictive power.

The processing involved adding new columns using different techniques which are:

- Using mathematical operations i.e. addition and division.

- Using the if-else conditional statements.

**Exploratory Data Analysis (EDA):**

This phase involves visualizing the key parameters that will enhance my model performance. These parameters include key features such as; total calls, total minutes, total charges and the target variable which is the churn.

Steps to follow include:

- Univariate Analysis

- Bi-variate Analysis

- Multivariate Analysis

**Univariate Analysis**

 ![image](https://github.com/user-attachments/assets/337750ba-2680-4365-b5e6-5647135267ab)

**Observations:**

- Normal Distribution: The histogram and KDE curve suggest that "Total Minutes" is approximately normally distributed. The bell shape is fairly symmetrical, and the KDE curve smooths this out nicely. This is a good sign for many modeling algorithms that assume normality

- Central Tendency: The distribution's peak (and thus the mean and median, given the symmetry) appears to be somewhere around 600-650 total minutes. This tells you the "average" total minutes used by your customers.

- Spread/Variability: The distribution shows a reasonable amount of spread or variability around the mean. Most customers fall within the range of roughly 400 to 800 total minutes. This spread is important; if the data was too tightly packed, it might mean the feature isn't very discriminative for churn.

**Bi-variate Analysis**

![image](https://github.com/user-attachments/assets/c7e928fc-e2f5-4ebe-b06b-260302d93f91)

**Observations:**

- Higher Average Charges for Churned Customers: The bar chart indicates that, on average, customers who have churned ("True") tend to have higher total charges than customers who haven't churned ("False").

- Potential for Predictive Power: This difference in average total charges suggests that "Total Charges" could be a useful feature for predicting churn. Customers with higher total charges might be more likely to churn.

**Multi-variate Analysis:**

![image](https://github.com/user-attachments/assets/1cf2306c-c85d-46b5-8dc3-6bc2ad2e2acd)

**Observations:**

- Strong Positive Correlation Between Total Minutes and Total Charges: The most prominent feature is the strong positive correlation (0.89) between Total Minutes and Total Charges. This is represented by the intense red color in that cell. It confirms what we saw in the scatter plot: customers who use more minutes tend to have higher charges.

- Weak Positive Correlation Between Total Calls and Total Minutes/Charges: The correlations between Total Calls and both Total Minutes (0.018) and Total Charges (0.022) are very weak and positive. The blue color of these cells indicates the near-zero correlation. This aligns with our earlier observation that the number of calls alone doesn't strongly predict total minutes or charges.

- No Multicollinearity Issues: While Total Minutes and Total Charges are highly correlated, this is not necessarily a Multicollinearity problem in the typical sense. Multicollinearity usually refers to having multiple predictor variables that are highly correlated. In this case, we are just examining the relationships between key usage metrics, and the high correlation between minutes and charges is expected.


**MODELING:**

My main aim was to develop multiple models and assess the best model for this classification problem based on the performance of each model. To achieve this the steps are as follows:

**Models to build:**

- Logistic regression model

- Decision tree classifier

- Random Forest Classifier

- XGBoost Classifier

To improve some of the models' performance some of the techniques  used include:

- Synthetic Minority Oversampling Technique-SMOTE (Handles class imbalance)

- Class weighting with scikit learn(Handles class imbalance)

- Hyper parameter using Grid Search and Randomized Search(Improves the model performance based on the key metrics such as recall)

**EVALUATION:**

For evaluation process, the evaluation metrics are:

- Recall(priority): It indicates what percentage of the classes to be analyzed were actually captured by the model.

- F1-score: Harmonic mean of precision and recall

- Precision: Measures how many predicted positives are actually positive

- Accuracy: Proportion of correctly classified instances out of the total

Visualizations to be used for the model analysis:

- **Confusion matrix:** It is a table used to evaluate the performance of a classification model by comparing predicted labels against actual (true) labels. It provides a detailed breakdown of correct and incorrect predictions, making it a valuable tool for understanding model performance.

- **ROC and AUC curves:** 

**Receiver Operator Characteristic Curve:** Illustrates the true positive rate against the false positive rate of the classifier 

**Area Under the Curve:** Singular value summarizing the ROC curve. A higher AUC indicates better performance


**Metrics Evaluation**


![image](https://github.com/user-attachments/assets/5f348582-164e-4a26-9bda-3415fa2466ff)

**Analysis of the Performance Metrics Comparison**

- Accuracy: The Tuned Random Forest and XGBoost models achieve the highest accuracy, significantly outperforming the Tuned SMOTE Logistic Regression model. The Balanced Decision Tree is slightly lower in accuracy but still relatively strong.

- Precision: XGBoost and Tuned Random Forest show the highest precision, indicating they make fewer false positive predictions. The Balanced Decision Tree also performs well, but the Tuned SMOTE Logistic Regression lags significantly, meaning it produces more false positives.

- Recall: All models have comparable recall, but the Balanced Decision Tree and Tuned Random Forest show a slight edge. High recall means the models are effectively identifying actual churn cases.

- F1-score: XGBoost and Tuned Random Forest have the highest F1-scores, balancing both precision and recall. The Balanced Decision Tree follows closely, while the Tuned SMOTE Logistic Regression lags behind due to its poor precision.

![image](https://github.com/user-attachments/assets/fd803e34-59e3-4e09-b54d-65c21d8a3216)

**Analysis based on the ROC Curve/AUC curve**

**Area Under the Curve (AUC) Interpretation:**

- Tuned Random Forest (AUC = 0.94): Best performing model, indicating a strong ability to distinguish between churners and non-churners.

- Balanced Decision Tree (AUC = 0.92): Slightly lower than Random Forest but still very effective.

- XGBoost (AUC = 0.91): Close to the Decision Tree and Random Forest, showing strong predictive power.

- Tuned SMOTE Logistic Regression (AUC = 0.85): Performs the worst among the models, but still better than random guessing (AUC = 0.5).

**Shape of the Curves:**

The Random Forest (green curve) and Balanced Decision Tree (orange curve) maintain the highest true positive rate while minimizing false positives.
XGBoost (red curve) also follows a strong pattern but is slightly below Random Forest. SMOTE Logistic Regression (blue curve) struggles with higher false positive rates compared to the other models.

**BEST MODEL:**

Since our business priority is identifying churned customers, the best model would be the one that maximizes recall while maintaining a good balance with precision. Therefore the best model is the Tuned Random Forest.

**Reasons for this choice:**

- Highest recall (captures the most churned customers) → From the performance metrics bar graphs, Tuned Random Forest has one of the highest recall scores.

- Highest AUC (0.94) → This means it effectively differentiates churners from non-churners.

- Good precision → While recall is the main goal, precision ensures that we don't flood the business with too many false churn predictions.

**Random Forest Feature Importance Analysis:**

![image](https://github.com/user-attachments/assets/1a0ed139-efff-473f-a522-7476bf6b9eef)

**Key Observations**

- Dominant Feature: Total Charges is overwhelmingly the most important feature, dwarfing all others. This indicates it's the strongest predictor in the model.

-Moderate Influence: Customer_Service_Calls, International Plan, Total_Day_Charge, Total Minutes, and Total_Day_Minutes have a moderate level of influence, though considerably less than Total Charges.

- Low to Negligible Influence: The remaining features have very low to negligible importance, suggesting minimal impact on the model's predictions.

**FINAL CONCLUSIONS:**

**Key Findings:**

- **Best Model:** The Tuned Random Forest model achieved the highest recall, ensuring that the majority of churned customers are correctly identified.

- **Feature Importance:** The Total Charges feature, engineered during preprocessing, played a crucial role in predicting churn across all models, highlighting its significance in customer behavior analysis.

- **Evaluation Metrics:** The Tuned Random Forest model demonstrated a strong balance of precision and recall, ensuring effective churn detection while minimizing false positives. It also achieved the highest AUC (0.94) in the ROC curve, confirming its superior ability to distinguish between churners and non-churners.

- **Business Impact:** This model will help SyriaTel identify at-risk customers early, allowing the company to take proactive retention measures such as personalized offers and improved customer service interventions.

**RECOMMENDATIONS:**

- Based on our findings and the business objective of reducing customer churn, we recommend the following actions:

1. **Customer Retention Strategy**

Focus on high total charge customers by offering discounts, loyalty rewards, or personalized service plans to prevent churn. In addition we can also monitor customers with low service usage and engage them through targeted offers and communication.

2. **Operational Implementation**

Deploy the Tuned Random Forest Model to predict churn regularly and generate reports for the customer retention team. Automate churn predictions and integrate insights into the company's CRM for proactive interventions.

3. **Future Improvements**

- Continuously update the model with new data to improve accuracy over time.

- Explore additional customer behavior data (e.g., complaints, network issues) to enhance feature engineering.

- Consider testing hybrid models combining Random Forest and XGBoost for potential performance gains.


 By implementing these recommendations, SyriaTel can leverage machine learning to reduce churn, enhance customer satisfaction, and improve long-term profitability.




