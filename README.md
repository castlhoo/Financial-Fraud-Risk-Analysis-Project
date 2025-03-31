# Financial-Fraud-Risk-Analysis-Project From kaggle

## 📌 Project Overview
### Detecting Accounting Fraud to Enhance Transactional Integrity
- This project analyzes the correlation between transaction amount (`Amount`), login behavior (`Login_Frequency`), and fraud occurrence (`Risk_Incident`).
- The insights aim to support fraud prevention systems by offering visual warnings through the UI or real-time monitoring mechanisms.

---

## 🧹 Data Preprocessing
We begin by loading and inspecting the dataset:
```python
import pandas as pd
PATH = "accounting_dataset.csv"
doc = pd.read_csv(PATH)
doc.head(20)
```

### Key Columns:
- `Transaction_ID`, `Date`, `Amount`, `Login_Frequency`, `Risk_Incident`, `System_Latency` among others.

### Basic EDA (Exploratory Data Analysis):
```python
doc.shape       # (10000, 18) records and columns
doc.describe()  # Summary statistics for numerical variables
```
![image](https://github.com/user-attachments/assets/b6c3e68d-4e39-441a-b8cf-49ee3b4c71b2)
![image](https://github.com/user-attachments/assets/3a1d43d4-fac2-43ef-b110-3e7413c40de3)

### Exporting Essential Columns to CSV and Date Preprocessing:
This helps modularize analysis and simplifies merging later.
```python
amount = doc[["Transaction_ID", "Amount"]]
amount.isnull().sum() 
amount.head()
amount.to_csv("amount.csv", index=False) 

login_frequency = doc[["Transaction_ID", "Login_Frequency"]]
login_frequency.isnull().sum() 
login_frequency.head()
login_frequency.to_csv("login_frequency.csv", index=False) 

risk_incident = doc[["Transaction_ID", "Risk_Incident"]]
risk_incident.isnull().sum() 
risk_incident.head()
risk_incident.to_csv("risk_incident.csv", index=False)
```

### Check Duplication
```python
amount.duplicated() # Nothing

login_frequency.duplicated() # Nothing

risk_incidents.duplicated() # 10001 value, which is unnecessary. REMOVE
risk_incident = risk_incident.drop(["Risk_Incident"])
risk_incident.tail()
```
---

## 💰 Relationship Between Transaction Amount and Risk_Incident
### Step 1: Merge and Create Quartiles
We merge the amount and incident datasets and segment amounts into quartiles.
```python
amount_incident = pd.merge(amount, risk_incident, on="Transaction_ID")
amount_incident["Amount_Q"] = pd.qcut(amount_incident["Amount"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
```

### Step 2: Compute Risk Rates by Quartile
This summarizes how fraud frequency changes across transaction amount groups.
```python
incident_summary = amount_incident.groupby("Amount_Q")["Risk_Incident"].agg(
    Risk_Count="sum", Total="count", Risk_Rate="mean").reset_index()
print(incident_summary)
```
![image](https://github.com/user-attachments/assets/03285d9d-6d8e-4314-87e6-670bc20f25a8)

### Step 3: Visualize with Plotly Bar Chart
We highlight Q4 if it shows the highest fraud rate.
```python
import plotly.graph_objects as go
colors = ['#1B80BF'] * 4
colors[3] = '#BF2C47'  # Emphasize Q4

fig = go.Figure(go.Bar(
    x=incident_summary["Amount_Q"],
    y=incident_summary["Risk_Rate"],
    text=incident_summary["Risk_Rate"],
    textposition="auto",
    marker_color=colors
))

fig.update_layout(
    title=dict(text="<b>Risk Incident Rate by Amount Quartile<b>", x=0.5, font=dict(size=15)),
    xaxis=dict(title="Amount Quartile", tickfont=dict(size=12)),
    yaxis=dict(title="Risk Rate", range=[0, 0.2]),
    template='plotly_white'
)

fig.add_annotation(
    x="Q4", y=0.16, text="<b>Increase Risk Rate</b>", showarrow=True,
    arrowhead=2, arrowwidth=2, arrowcolor="#77BDD9",
    font=dict(size=10, color="#ffffff"), bgcolor="#F22E62", opacity=0.8
)
fig.show()
```
![image](https://github.com/user-attachments/assets/1f086c97-18d4-48d9-b514-d75dafd0642f)

> 🔍 **Insight**: The bar chart illustrates that transactions in the highest quartile (Q4) exhibit a noticeably higher fraud rate (15.32%) compared to lower quartiles. While Q1 through Q3 remain relatively stable around 14.2%, Q4 stands out with an upward shift, indicating that high-value transactions are more prone to fraud incidents. This pattern suggests that enhanced scrutiny and controls should be prioritized for transactions in the upper quartile.

---

## 📅 Monthly Heatmap by Amount Decile
We explore how fraud risk changes over time across 10 quantiles of transaction amounts.

### Step 1: Segment Amount and Extract Year-Month
```python
amount_incident_Q2 = amount_incident.copy()
amount_incident_Q2["Amount_Q"] = pd.qcut(amount_incident_Q2["Amount"], q=10, labels=["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"])

date = doc[["Transaction_ID", "Date"]]
date["Date"] = pd.to_datetime(date["Date"])
date.dropna(inplace=True)
date["YearMonth"] = date["Date"].dt.to_period("M")

monthly_q_summary = pd.merge(amount_incident_Q2, date, on="Transaction_ID")
```
![image](https://github.com/user-attachments/assets/8e6c3ed2-0029-42ff-a3a2-eaf9c313e83e)
![image](https://github.com/user-attachments/assets/854914a0-e4f3-4d15-9910-b1654e378c03)

### Step 2: Create Pivot Table & Heatmap
```python
pivot_df = monthly_q_summary.pivot_table(
    index="Amount_Q", columns="YearMonth", values="Risk_Incident", aggfunc="mean").sort_index()

pivot_df.index = pivot_df.index.astype(str)
pivot_df.columns = pivot_df.columns.astype(str)

fig = go.Figure(go.Heatmap(
    z=pivot_df.values,
    x=pivot_df.columns,
    y=pivot_df.index,
    zmin=0,
    zmax=0.2,
    colorscale='Reds',
    text=pivot_df.round(3).values,
    texttemplate="%{text}",
    hovertemplate="Month: %{x}<br>Quantile: %{y}<br>Risk: %{z:.3f}<extra></extra>"
))

fig.update_layout(
    title=dict(text="Monthly Risk Rate by Amount Quantile", x=0.5, font=dict(size=16)),
    xaxis=dict(title="Year-Month"),
    yaxis=dict(title="Amount Quantile"),
    margin=dict(t=50, b=50, l=70, r=20)
)
fig.show()
```
![image](https://github.com/user-attachments/assets/d926272f-5eef-437c-b71a-48deec8c77a0)

> 🔍 **Insight**: The heatmap shows that higher transaction deciles (especially Q9 and Q10) consistently exhibit elevated fraud rates across multiple months. Notably, Q10 peaks in January, May, and November 2024, suggesting periods of heightened risk in high-value transactions. In contrast, lower deciles (Q1–Q4) maintain relatively lower and more stable fraud rates, reinforcing the pattern that fraud likelihood scales with transaction size.

---

## 🔐 Login Frequency vs. Risk_Incident
We analyze whether login behavior correlates with fraud likelihood.

### Step 1: Merge, Describe and Correlation
```python
login_incident = pd.merge(login_frequency, risk_incident, on="Transaction_ID")
login_incident.describe()

corr = login_incident.corr(numeric_only = True)
print(corr) # negative correlation
```
![image](https://github.com/user-attachments/assets/d1dc04e6-4f19-48ee-85ca-bcc0d452ea32)
![image](https://github.com/user-attachments/assets/73ccc0b8-d4e9-4b5d-ba1e-4c36c6593841)


### Step 2: Aggregate by Login Frequency
```python
incident_summary2 = login_incident.groupby("Login_Frequency")["Risk_Incident"].agg(
    Risk_count="sum", Total="count", Risk_Rate="mean").reset_index()
print(incident_summary2)
```
![image](https://github.com/user-attachments/assets/11212baf-2a9b-4d05-9cf9-078d38090c4e)

### Step 3: Visualize
```python
colors = ['#1B80BF'] * len(incident_summary2)
colors[2] = '#BF2C47'  # Highlight Login_Frequency = 3

fig = go.Figure(go.Bar(
    x=incident_summary2["Login_Frequency"],
    y=incident_summary2["Risk_Rate"],
    text=incident_summary2["Risk_Rate"],
    textposition="auto",
    texttemplate="%{text:.2f}",
    marker_color=colors
))

fig.update_layout(
    title=dict(text="<b>Risk Incident Rate by Login Frequency<b>", x=0.5, font=dict(size=15)),
    xaxis=dict(title="Login Frequency", dtick=1, tickfont=dict(size=12)),
    yaxis=dict(title="Risk Rate", range=[0, 0.2]),
    template='plotly_white'
)

fig.add_annotation(
    x=3, y=0.17, text="<b>The Most Largest</b>", showarrow=True,
    arrowhead=2, arrowwidth=2, arrowcolor="#77BDD9",
    font=dict(size=10, color="#ffffff"), bgcolor="#F22E62", opacity=0.8
)
fig.show()
```
![image](https://github.com/user-attachments/assets/b5ec23a9-704e-4ff8-aa58-5ec178015f26)

> 🔍 **Insight**: The chart reveals a distinct peak in fraud risk for users with a login frequency of 3, where the fraud rate reaches 17%. Overall, accounts with lower login activity (1–4 times) show elevated fraud rates compared to more active users. This suggests that infrequently accessed accounts may be more vulnerable—potentially due to account dormancy or reduced user vigilance.

---

## 🧠 Summary of Insights
- 💰 Fraud tends to increase with higher transaction amounts, particularly in Q8–Q10.
- 📅 Certain months, such as January and November, experience noticeable spikes in fraud activity.
- 🔐 Accounts with low login frequency (especially 3 logins or fewer) show significantly higher fraud rates.

---

## 🔮 Future Directions
- Add behavioral features such as device type or login location to enrich analysis.
- Train classification models (e.g., Random Forest, Logistic Regression) to detect high-risk transactions.
- Develop dashboards to monitor fraud trends in real time and alert on anomalies.

---
**Author**: Seongho Kim  
**Project**: Financial Transaction and Risk Management Dataset Analysis
**Data** From https://www.kaggle.com/datasets/ziya07/financial-transaction-and-risk-management-dataset/data (Kaggle)
