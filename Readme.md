# 🔐 SecureGuard  
### AI-Powered Insurance Fraud Detection & Risk Triage System (MVP)

SecureGuard is a **machine learning–driven fraud detection system** designed to **identify, score, and triage suspicious insurance claims**.  
It leverages **advanced feature engineering**, **unsupervised learning**, and **ensemble modeling** to produce an **actionable investigator dashboard**.

---

## 🚨 Problem Statement

Insurance fraud leads to:

- 💸 Significant financial losses  
- 🕵️ Heavy manual investigation workload  
- ⏳ Slow claim processing  

Traditional rule-based systems:

- ❌ Fail to adapt to new fraud patterns  
- ❌ Offer poor explainability  
- ❌ Do not scale efficiently  

**SecureGuard solves this by assigning a real-time fraud risk score and recommended action to every claim.**

---

## 🚀 Solution Overview

SecureGuard:

- 📊 Learns fraud patterns from historical claims data  
- 🔢 Generates **fraud risk scores (0–100)**  
- 🚩 Flags **high-risk claims** for investigation  
- ✅ Auto-approves **low-risk claims**  
- 📁 Produces a **ready-to-use CSV dashboard** for investigators  

---

## 🧠 Key Features

### 🔬 Advanced Feature Engineering
- Missing-value anomaly detection (e.g., `Age = 0`)  
- Customer behavior clustering using **K-Means**  
- High-risk interaction features  
- Frequency encoding for categorical variables  

### 🤖 Hybrid Learning Approach
- **Supervised learning** for fraud classification  
- **Unsupervised learning** for customer profiling  

### ⚖️ Class Imbalance Handling
- Fraud-to-legitimate claim ratio balanced to **1:3**  
- Reduces bias toward the majority class  

### 📌 Action-Oriented Output
Each claim includes:
- **Fraud Risk Score**
- **Recommended Action**
- **Human-readable Red Flags**

---

## 🏗️ System Architecture

Raw Claims Data
↓
Feature Engineering
↓
Customer Clustering (K-Means)
↓
Balanced Training Dataset
↓
Ensemble Model (Random Forest)
↓
Risk Scoring Engine
↓
Investigator Dashboard (CSV)


---

## 📊 Risk Classification Logic

| Risk Score | Action |
|-----------|--------|
| ≥ 25 | 🚨 INVESTIGATE |
| < 25 | ✅ AUTO-APPROVE |

> Threshold intentionally set low for **high fraud sensitivity** in MVP.

---

## 🧪 Models & Techniques Used

- 🌲 Random Forest Classifier  
- 🧩 K-Means Clustering  
- 🏷️ Label Encoding  
- 📈 Frequency Encoding  
- 🔁 Downsampling for class balancing  

---

## 📁 Project Structure

├── secureguard.py
│ └─ Core ML pipeline & MVP engine
├── SecureGuard1.ipynb
│ └─ Notebook for experiments & demo
├── SecureGuard_MVP_Dashboard.csv
│ └─ Generated investigator dashboard
└── README.md


## ⚙️ How to Run

### 1️⃣ Install Dependencies
pip install pandas numpy scikit-learn
2️⃣ Run the System


python secureguard.py
3️⃣ Output
🖥️ Console displays highest-risk claims

📄 CSV generated: SecureGuard_MVP_Dashboard.csv

📋 Dashboard Columns
Column	Description
Risk_Score	Fraud probability (0–100)
Action	Investigate / Auto-Approve
Red_Flags	AI-generated explanation

🚩 Example Red Flags
Missing Age Data

High Deductible Anomaly

Policy Holder at Fault

Routine Claim

🎯 Use Cases
🏦 Insurance companies

🕵️ Claims investigation teams

📊 Fraud analytics research

🎓 Academic ML projects

🤝 AI decision-support systems

🔮 Future Enhancements
SHAP-based explainability

Real-time API integration

Deep learning models

Investigator feedback loop

Web-based dashboard (Streamlit / React)

👨‍💻 Author
SecureGuard MVP
Machine Learning–Driven Fraud Detection System

📜 License
📘 Released for educational and research purposes only.

⭐ If you find this project useful, consider starring the repository.


