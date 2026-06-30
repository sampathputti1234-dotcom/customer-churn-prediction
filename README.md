# customer-churn-prediction
Predicting e-commerce customer churn using Python, Pandas, Seaborn, and Scikit-Learn to deliver actionable business insights

*****************************************************************************************************************************

E-Commerce Customer Retention & Churn Prediction
📌 Project Overview
Losing customers (churn) is one of the most expensive problems an e-commerce platform can face. This project performs an end-to-end Exploratory Data Analysis (EDA) and builds a predictive Machine Learning classification model to discover why customers leave and identify high-risk accounts before they churn.

By analyzing customer behavior patterns, this project provides data-driven, actionable insights that can help business stakeholders design proactive marketing and retention strategies.

*****************************************************************************************************************************
🛠️ Tech Stack & Libraries
*****************************************************************************************************************************
Language:- Python

Data Manipulation:- Pandas, NumPy

Data Visualization:- Seaborn, Matplotlib

Machine Learning:- Scikit-Learn (Logistic Regression, Train-Test Split, StandardScaler, Confusion Matrix Metrics)

******************************************************************************************************************************************

📊 Key Insights from Exploratory Data Analysis (EDA)
During the EDA process, several key factors were uncovered that correlate strongly with customer churn:

The Distance Dilemma:- Customers who live further away from the main warehouse exhibit significantly higher churn rates. This suggests that shipping transit times or local delivery challenges may be negatively impacting customer satisfaction.

Customer Friction points:- A massive spike in churn was visible among customers who had raised official complaints. This indicates that customer service response times and resolution quality are critical failure points in the retention pipeline.

Data Quality Win:- Cleaned structural inconsistencies in categorical data (e.g., standardizing text inputs like merging typos from Mobile Phone into Mobile), ensuring a highly accurate baseline dataset for analysis.

******************************************************************************************************************************************

🤖 Machine Learning Approach & Best Practices
To transition from retrospective analysis to proactive prediction, a Logistic Regression classifier was trained to flag potential churned accounts.

Implementation Highlights:-
Data Leakage Mitigation: The dataset was split into training and testing segments before processing feature means. This guarantees that evaluation scores accurately represent real-world predictive performance.

Feature Scaling:- Implemented StandardScaler to normalize numeric ranges across variations like CashbackAmount and binary flags like Complain, preventing weight distortion during gradient updates.

Stratified Splitting:- Utilized stratified=y on the target class during initialization to perfectly balance the churn-to-retained ratio across validation boundaries.

**********************************************************************************************************************************************

Performance Evaluation
***********************
The evaluation utilizes both an Accuracy Score and a Seaborn Confusion Matrix Heatmap to visually assess true positives and map where the baseline model struggles with false classifications.

