# HR DATA ANALYSIS – MACHINE LEARNING PROJECT (PYTHON)

A complete machine learning workflow built using Python & Google Colab to analyze HR hiring data and predict whether a candidate will **join** or **not join** after receiving an offer.  
This project includes **EDA**, **Data Cleaning**, **Feature Engineering**, **SMOTE balancing**, **Logistic Regression modeling**, and **Model Evaluation**.

---

## 📌 1. Project Overview

Many organizations face uncertainty regarding candidate joining behavior after extending job offers.  
This project builds a logistic regression model that predicts the likelihood of a candidate joining.
- Expected Hike vs Offered Hike  
- Location & Band  
- Recruiting channel  
- Experience & Age  
- Acceptance Delay  
- Bonus / Relocation  
- Notice period 

### ✔ Objectives
- Explore HR data and identify joining patterns  
- Clean & preprocess the dataset  
- Apply SMOTE to handle class imbalance  
- Implement Logistic Regression model  
- Evaluate performance using Confusion Matrix & ROC Curve  
- Provide a reusable end-to-end ML pipeline

---

## 📂 2. Dataset Information

**Dataset Filename:** `IMB553-XLS-ENG.xlsx`  

The dataset includes information related to:
- Candidate demographics  
- Expected vs offered compensation  
- Location & line of business  
- Experience & age  
- Offer acceptance behavior  

---

## 📊 3. Sample Data

> 
Below is a preview of the HR dataset used in this project:

| Candidate.Ref | DOJ.Extended | Duration.to.accept.offer | Notice.period | Offered.band | Pecent.hike.expected.in.CTC | Percent.hike.offered.in.CTC | Percent.difference.CTC | Joining.Bonus | Candidate.relocate.actual | Gender | Candidate.Source     | Rex.in.Yrs | LOB       | Location | Age | Status      |
|---------------|--------------|---------------------------|----------------|---------------|------------------------------|------------------------------|--------------------------|----------------|----------------------------|--------|------------------------|-------------|-----------|----------|------|-------------|
| 2110407       | Yes          | 14                        | 30             | E2            | -20.79                      | 13.16                       | 42.86                   | No             | No                         | Female | Agency                | 7           | ERS       | Noida    | 34   | Joined      |
| 2112635       | No           | 18                        | 30             | E2            | 50                         | 320                        | 180                    | No             | No                         | Male   | Employee Referral     | 8           | INFRA     | Chennai  | 34   | Joined      |
| 2112838       | No           | 3                         | 45             | E2            | 42.84                       | 42.84                       | 0                      | No             | No                         | Male   | Agency                | 4           | INFRA     | Noida    | 27   | Joined      |
| 2115021       | No           | 26                        | 30             | E2            | 42.84                       | 42.84                       | 0                      | No             | No                         | Male   | Employee Referral     | 4           | INFRA     | Noida    | 34   | Joined      |
| 2115125       | Yes          | 1                         | 120            | E2            | 42.59                       | 42.59                       | 0                      | No             | Yes                        | Male   | Employee Referral     | 6           | INFRA     | Noida    | 34   | Joined      |
| 2117167       | Yes          | 17                        | 30             | E1            | 42.83                       | 42.83                       | 0                      | No             | No                         | Male   | Employee Referral     | 2           | INFRA     | Noida    | 34   | Joined      |
| 2119124       | Yes          | 37                        | 30             | E2            | 31.58                       | 31.58                       | 0                      | No             | No                         | Male   | Employee Referral     | 7           | INFRA     | Noida    | 32   | Joined      |
| 2121918       | No           | —                         | 45             | E2            | 40                         | 208.64                      | 120.45                 | No             | No                         | Male   | Employee Referral     | 4           | INFRA     | Noida    | 34   | Not Joined  |
| 2127572       | Yes          | 16                        | 0              | E1            | -20                         | -20                         | 0                      | No             | No                         | Female | Direct                | 8           | Healthcare| Noida    | 34   | Joined      |
| 2137866       | No           | —                         | 30             | E1            | 1                          | -51.37                      | -55.71                 | No             | No                         | Male   | Direct                | 4           | ERS       | Noida    | 34   | Not Joined  |
| 2138169       | No           | 1                         | 30             | E1            | -22.22                     | -22.22                      | 0                      | No             | No                         | Female | Employee Referral     | 3           | BFSI      | Gurgaon  | 26   | Joined      |
| 2143362       | No           | 6                         | 30             | E1            | 240                        | 220                        | -5.88                  | No             | No                         | Male   | Employee Referral     | 3           | CSMP      | Chennai  | 34   | Joined      |
| 2151180       | No           | 120                       | 30             | E2            | 5.26                        | -60.53                      | -62.5                  | No             | No                         | Male   | Employee Referral     | 3           | INFRA     | Noida    | 34   | Not Joined  |
| 2154264       | No           | 3                         | 0              | E2            | 28.21                       | 37.18                       | 7                      | No             | No                         | Male   | Employee Referral     | 7           | INFRA     | Chennai  | 34   | Joined      |
| 2156236       | Yes          | 14                        | 30             | E2            | 50                         | 287.5                       | 158.33                 | No             | No                         | Male   | Agency                | 7           | INFRA     | Noida    | 29   | Not Joined  |
| 2158703       | No           | 44                        | 75             | E2            | 45.45                       | 60                          | 10                     | No             | No                         | Male   | Direct                | 8           | INFRA     | Noida    | 34   | Not Joined  |
| 2161257       | No           | 7                         | 30             | E3            | 53.85                       | 50                          | -2.5                   | No             | No                         | Male   | Employee Referral     | 5           | INFRA     | Noida    | 34   | Not Joined  |




---

## 📘 4. Data Dictionary

| Column Name | Description |
|-------------|-------------|
| `Candidate.Ref` | Unique candidate reference number |
| `DOJ.Extended` | Whether joining date was extended |
| `Duration.to.accept.offer` | Number of days taken to accept offer |
| `Notice.period` | Candidate’s notice period |
| `Offered.band` | Job band offered (E1, E2, …) |
| `Percent.hike.expected.in.CTC` | Expected salary hike |
| `Percent.hike.offered.in.CTC` | Hike offered by company |
| `Joining.Bonus` | Whether a joining bonus was offered |
| `Candidate.relocate.actual` | Whether the candidate agreed to relocate |
| `Gender` | Male / Female |
| `Candidate.Source` | Employee Referral / Direct / Agency |
| `Rex.in.Yrs` | Relevant years of experience |
| `LOB` | Line of business |
| `Location` | City of posting |
| `Age` | Candidate age |
| `Status` | Joined / Not Joined |

---

## 🛠 4. End-to-End Modeling Steps (Detailed)

This section documents every technical step performed in the Jupyter/Colab notebook.

---

### 🔹 Step 1 — Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
```
####  ✔ Ignore Warnings (Clean Notebook View)
```python
import warnings
warnings.filterwarnings('ignore')
```
---
### 🔹 Step 2 — Load Dataset
```python
df = pd.read_excel("IMB553-XLS-ENG.xlsx")
df.head()
```
<img width="1271" height="172" alt="image" src="https://github.com/user-attachments/assets/34ab6122-2ebd-4ddb-9b09-15ed28fe3644" />



---
### 🔹 Step 3 — Data Cleaning

Cleaning includes:

Removing nulls

Removing duplicates

Handling inconsistent data

####  ✔ Fixing wrong types
```python
df.isnull().sum()
df.dropna(inplace=True)
```
####  ✔ After dropping nulls,, re-check shape:
```python
df.shape
```
####  ✔ Check Data Types & Columns
```python
df.dtypes
df.columns
```
####  ✔ Rename Columns
```python
df = df.rename(columns={
    'Candidate.Ref':'Candidate_ref',
    'DOJ.Extended':'DOJ_extended',
    'Duration.to.accept.offer':'Days_accept',
    'Notice.period':'Notice',
    'Offered.band':'Band',
    'Pecent.hike.expected.in.CTC':'Expected_hike',
    'Percent.hike.offered.in.CTC':'Offered_hike',
    'Percent.difference.CTC':'Difference_hike',
    'Joining.Bonus':'JoinBonus',
    'Candidate.relocate.actual':'Relocated',
    'Candidate.Source':'Source',
    'Rex.in.Yrs':'Rex'
})
```
####  ✔ Rearrange Columns
```python
df = df[['Candidate_ref', 'DOJ_extended', 'Days_accept', 'Notice', 'Band',
         'Expected_hike','Offered_hike','Difference_hike','JoinBonus',
         'Relocated','Gender','Source','Rex','LOB','Location','Age','Status']]
```
<img width="1270" height="212" alt="image" src="https://github.com/user-attachments/assets/5ed795fe-0b15-4004-a9f1-c244a026a40c" />



---
### 🔹 Step 4 — Exploratory Data Analysis (EDA)

Performed:

Count plots

Distribution plots

KDE plots

Pairplots

Correlation heatmap
```python
import seaborn as sns
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(cat):
    sns.countplot(x=df[col], ax=axes[i])

plt.tight_layout()
plt.show()
```
<img width="500" height="600" alt="image" src="https://github.com/user-attachments/assets/020e9d5b-358d-423c-b3aa-c4b3c709e537" />

----
### 🔹 Step 5 — Data Preprocessing 
```python
df.drop('Candidate_ref', axis=1, inplace=True)
```
####  ✔ Extracting Dependent and Dependent variables
```python
x = df1.iloc[:,:-1]
y = df1.iloc[:,-1]
```
####  ✔ Handlining categorical values
```python
y=df["Status"].map({"Joined":0,"Not Joined":1})
cato=['DOJ_extended', 'Offered_band', 'Joining_Bonus',
       'Relocated', 'Gender', 'Source', 'LOB', 'Location']
x = pd.get_dummies(x,columns=cato, drop_first=True)
```
####  ✔ Apply SMOTE for Balancing
```python
x_res,y_res=SMOTE(k_neighbors=3).fit_resample(x,y)
```

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/ca921817-54b9-45cc-b6b4-abdbf7c9ad0b" />




####  ✔ Standard Scaling
```python
x_col=['Accept_duration', 'Notice_period', 'Percent_hike_expected',
       'Percent_hike_offered', 'Percent_difference', 'Rex', 'Age',
       'DOJ_extended_Yes', 'Offered_band_E1', 'Offered_band_E2',
       'Offered_band_E3', 'Joining_Bonus_Yes', 'Relocated_Yes', 'Gender_Male',
       'Source_Direct', 'Source_Employee Referral', 'LOB_BFSI', 'LOB_CSMP',
       'LOB_EAS', 'LOB_ERS', 'LOB_ETS', 'LOB_Healthcare', 'LOB_INFRA',
       'LOB_MMS', 'Location_Bangalore', 'Location_Chennai', 'Location_Cochin',
       'Location_Gurgaon', 'Location_Hyderabad', 'Location_Kolkata',
       'Location_Mumbai', 'Location_Noida', 'Location_Others',
       'Location_Pune']
for i in x_col:
 sr = StandardScaler()
 x_res[i]=sr.fit_transform(x_res[[i]])
x_res
```
---
### 🔹 Step 6 — Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42)
)
```
---
### 🔹 Step 7 — Logistic Regression Model(Model Selection)
#### ✔ Fit Model
```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
```
#### ✔ Model Accuracy
```python
lr.score(X_train, y_train), lr.score(X_test, y_test)
```
#### ✔ Predictions
```python
y_pred = lr.predict(X_test)
```
---
### 🔹 Step 8 — Matrix
#### ✔ Accuracy Score
```python
accuracy_score(y_test, y_pred)
```
#### ✔ Classification Report
```python
print(classification_report(y_test, y_pred))
```


<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/ead7dfd9-87d3-4a6f-bac8-a90582fc2594" />



#### ✔ Confusion Matrix
```python
confusion_matrix(y_test, y_pred)
```
Heatmap:
```python
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()
```


<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/9bcff465-9f73-4b9d-8774-1f6a7c27771c" />


### 🔹 Step 9 — ROC Curve & AUC Score
```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
```
#### ✔ Plot ROC Curve
```python
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], 'k--')
plt.legend()
plt.title("Logistic Regression ROC Curve")
plt.show()
```
 
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/75247110-8b8c-440d-91fb-e4d730361013" />



---
### 🔹 Step 10 — Coefficient & Intercept
```python
lr.coef_
lr.intercept_
```
coefficient

<img width="763" height="205" alt="image" src="https://github.com/user-attachments/assets/a9e5d6e4-361e-4717-9258-c2293a57288b" />



intercept

<img width="270" height="81" alt="image" src="https://github.com/user-attachments/assets/eb0ce3df-76b1-4c82-992d-4a109f0d33f1" />



---
### 🔹 Step 11 — Manual Logistic Regression Calculation (Optional)

Notebook computes predicted probability using formula:

(1 / (1 + e^(-x)))

<img width="1715" height="291" alt="image" src="https://github.com/user-attachments/assets/3051bef5-061b-4433-8078-a77c32953e7a" />

---
## 🎯 5. Conclusion
✔ AUC Score = 0.825

This indicates very good prediction capability.

✔ Key Takeaways:

SMOTE balanced the dataset effectively

Scaling improved model performance

Logistic Regression is interpretable and accurate

Top influencing factors include:

Expected vs Offered hike

Notice period

Relevant experience

Age

---


