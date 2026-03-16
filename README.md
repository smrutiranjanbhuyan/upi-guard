# 🛡 UPI-Guard++

## Graph-Aware, Cost-Sensitive UPI Fraud Detection System

UPI-Guard++ is a research-grade fraud detection system built using a real-world UPI fraud dataset sourced from Kaggle.  
It integrates temporal modeling, behavioral profiling, graph centrality features, and cost-sensitive learning into a deployable machine learning system.

---

## 🚀 Key Features

- 📊 UPI fraud dataset sourced from Kaggle
- 🧠 Graph-based fraud detection (PageRank + Degree Centrality)
- ⏱ Temporal velocity modeling (1-hour rolling window)
- 📈 Behavioral anomaly detection (Z-score)
- ⚖ Cost-sensitive threshold optimization
- 🤖 XGBoost ensemble model
- 🌐 FastAPI REST API
- 📊 Streamlit interactive dashboard
- 📉 ROC-AUC & PR-AUC evaluation

---

## 📂 Project Structure

```
upi-guard/
│
├── data/
│   └── upi_transactions.csv
│
├── model/
│   ├── fraud_model.pkl
│   ├── feature_columns.pkl
│   └── threshold.txt
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── optimize_threshold.py
│
├── api/
│   └── main.py
│
├── ui/
│   └── dashboard.py
│
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset Description

The project uses a **UPI transaction fraud detection dataset sourced from Kaggle**.

The dataset contains transaction records used for machine learning-based fraud detection.

### Features include:

- amount
- transaction_type
- sender_state
- receiver_state
- sender_bank
- receiver_bank
- device_type
- network_type
- account_age_days
- txn_velocity_1h
- graph centrality metrics
- fraud_flag (target variable)

---

## 🏗 System Architecture

```
Transaction Data
        ↓
Feature Engineering
        ↓
Graph Construction
        ↓
SMOTE Imbalance Handling
        ↓
XGBoost Model
        ↓
Cost-Sensitive Threshold
        ↓
API Deployment
        ↓
Interactive Dashboard
```

---

## ⚙ Installation

### 1️⃣ Clone Repository

```bash
git clone <your_repo_url>
cd upi-guard
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Train Model

```bash
cd src
python train.py
cd ..
```

After training, model files will be saved in:

```
model/fraud_model.pkl
model/feature_columns.pkl
```

---

## ⚖ Optimize Decision Threshold

```bash
cd src
python optimize_threshold.py
cd ..
```

---

## 🌐 Run FastAPI Backend

```bash
python -m uvicorn api.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 📊 Run Streamlit Dashboard

Open a new terminal:

```bash
.\venv\Scripts\Activate.ps1
python -m streamlit run ui/dashboard.py
```

Dashboard opens at:

```
http://localhost:8501
```

---

## 📈 Model Evaluation Metrics

The model is evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cost-based Financial Loss

Target performance:

- ROC-AUC: 0.94 – 0.98
- Fraud Recall: > 90%
- Precision: 85–93%

---

## 🧠 Research Contributions

- Graph-aware fraud detection
- Temporal velocity modeling
- Behavioral anomaly scoring
- Cost-sensitive learning framework
- Real-time deployment architecture

---

## 🎓 Interview Summary

Built a graph-enhanced, cost-sensitive UPI fraud detection system trained on a Kaggle UPI fraud dataset achieving ~97% ROC-AUC, deployed via FastAPI and Streamlit dashboard.

---

## 📌 Future Improvements

- SHAP explainability
- Graph Neural Networks (GNN)
- Real-time streaming fraud detection
- Distributed graph processing
- Production Docker deployment

---

## 🏆 Author

Smruti Ranjan Bhuyan

---

## 📜 License

This project is for academic and research purposes.