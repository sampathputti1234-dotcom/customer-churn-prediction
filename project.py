import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# --- STEP 1: DATA LOADING ---
df = pd.read_csv("C:/Users/Putti Sampath/Sampath/AI and ML/ml/archive (1)/data_ecommerce_customer_churn.csv")

# --- STEP 2: DATA CLEANING (TEXT STAGE) ---
# Fixing structural typos in text BEFORE doing any numeric splitting
typo_map = {'Mobile Phone': 'Mobile'}
df["PreferedOrderCat"] = df['PreferedOrderCat'].replace(typo_map)

# --- STEP 3: DATA VISUALIZATION (EDA) ---
sns.set_theme(style='whitegrid')
colors = sns.color_palette('pastel')

# Chart 1: Pie Chart
churn_counts = df['Churn'].value_counts()

def make_autopercentage(values):
    def my_autopercentage(per):
        total = sum(values)
        val = int(round(per * total / 100.0))
        return f'{per:.1f}%\n ({val:,})'
    return my_autopercentage
        
plt.figure(figsize=(6,6))
plt.pie(churn_counts,
        labels=['Retained', 'Churned'], # Fixed typo here from 'Chained'
        autopct=make_autopercentage(churn_counts),
        startangle=90,
        colors=[colors[0], colors[3]],
        wedgeprops=dict(width=0.5, edgecolor='w'))

plt.title('Overall Customer Churn Rate', fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Chart 2: Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='WarehouseToHome', data=df, palette=[colors[0], colors[3]])
plt.xticks([0,1], ['Retained', 'Churned'])
plt.xlabel('Customer Status', fontsize=12)
plt.ylabel("Distance from warehouse to Home (km)", fontsize=12)
plt.title("Does Warehouse Distance Drive Churn", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# Chart 3: Countplot
plt.figure(figsize=(8,5))
sns.countplot(x='Complain', hue='Churn', data=df, palette=[colors[0], colors[3]])
plt.xticks([0,1], ['No Complaint', 'Raised Complaint'])
plt.xlabel('Customer Complaints', fontsize=12)
plt.ylabel('Number of customers', fontsize=12)
plt.title('Analysis of Churn Based on Customer Complaints', fontsize=14, fontweight='bold', pad=15)
plt.legend(title='Status', labels=['Retained', 'Churned'])
plt.tight_layout()
plt.show()



# --- STEP 4: MACHINE LEARNING PIPELINE ---

# 1.Target and features
features = ['WarehouseToHome', 'Complain', 'DaySinceLastOrder', 'CashbackAmount']
X = df[features]
y = df['Churn']

# 2. Split Data FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3.The mean values from the training data ONLY
warehouse_mean = X_train['WarehouseToHome'].mean()
order_mean = X_train['DaySinceLastOrder'].mean()
cashback_mean = X_train['CashbackAmount'].mean() # 👈 Added this for CashbackAmount!

# 4.Filling the Missing data properly WITHOUT using inplace=True (Prevents slice errors)
X_train['WarehouseToHome'] = X_train['WarehouseToHome'].fillna(warehouse_mean)
X_test['WarehouseToHome'] = X_test['WarehouseToHome'].fillna(warehouse_mean)

X_train['DaySinceLastOrder'] = X_train['DaySinceLastOrder'].fillna(order_mean)
X_test['DaySinceLastOrder'] = X_test['DaySinceLastOrder'].fillna(order_mean)

X_train['CashbackAmount'] = X_train['CashbackAmount'].fillna(cashback_mean) # 👈 Added this!
X_test['CashbackAmount'] = X_test['CashbackAmount'].fillna(cashback_mean)   # 👈 Added this!

# 5. Feature Scaling (Now it will run perfectly!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 6.Training the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7.Evaluating the Model
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy Score: {accuracy * 100:.2f}%")

# 8.Plot Heatmap Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Retained', 'Predicted Churned'],
            yticklabels=['Actual Retained', 'Actual Churned'])

plt.title('Confusion Matrix for Churn Prediction', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Actual Status', fontsize=12)
plt.xlabel('Predicted Status', fontsize=12)
plt.tight_layout()
plt.show()