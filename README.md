# 📈 End-to-End Sales Forecasting & Demand Intelligence System

Welcome to the **Sales Forecasting & Demand Intelligence System**, built during the Data Science internship at **Xylofy AI**. 

This system provides a proactive approach to retail inventory management. By combining statistical time-series models, machine learning algorithms, anomaly detection, and clustering, it predicts future sales demand, identifies operational outliers, and segments products for tailored stocking strategies.

---

## 🔮 Model Performance & Validation
Trained and validated on monthly sales data. Models were evaluated using the last 6 months of historical data (July - December 2018):

| Model | MAE ($) | RMSE ($) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **SARIMA** (Best Model) | **$14,862.39** | **$17,299.71** | **17.89%** |
| **Facebook Prophet** | $14,309.99 | $18,954.58 | **17.47%** |
| **XGBoost Regressor** | $20,999.65 | $22,951.30 | 25.62% |

> [!NOTE]
> **SARIMA** yielded the lowest RMSE, making it the most robust forecasting backend for overall demand trends.

---

## 🚀 Key Insights & Features

* **Task 1: Analytical Questions**
  * **Highest Revenue Category:** Technology ($827,455.87) leads sales.
  * **Consistent Growth Region:** East Region exhibits the most stable year-over-year expansion (average growth ~18%, lowest volatility of 1.79%).
  * **Fulfillment Time:** 3.96 days average order-to-shipment time.
  * **Seasonality:** High Q4 peaks in November and December.
* **Task 5: Anomaly Detection**
  * Implements **Isolation Forest** and **Z-Score** models on weekly sales.
  * Captures pre-holiday promotional spikes (Black Friday / Cyber Monday) and mid-September B2B contracts.
* **Task 6: Product Segmentation**
  * Uses **K-Means Clustering** to segment sub-categories into 4 inventory groups: *High Volume Stable, Superstars (High Revenue/Volatile), Low Volume Stable, and Moderate Volume Volatile*.
* **Task 7: Streamlit Dashboard**
  * Multi-page app for interactive exploration of sales trends, segment forecasts, weekly anomalies, and demand clusters.

---

## 📂 Project Structure

```
├── analysis.ipynb         # Complete executed Jupyter Notebook
├── app.py                 # Multi-page Streamlit Dashboard app
├── summary.pdf            # 2-page PDF report for the CFO & Head of Supply Chain
├── requirements.txt       # Dependencies
├── product_segments.csv   # Cluster outputs
├── weekly_sales_anomalies.csv # Weekly anomalies data
├── train.csv              # Superstore Sales dataset
├── vgsales.csv            # Supplementary Video Games Sales dataset
└── charts/                # Generated PNG visualizations
```

---

## 🛠️ How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
