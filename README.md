# Supply Chain Risk Intelligence & Delivery Performance Prediction

## Project Overview

This project explores supply chain delivery performance, identifies potential delivery risks, and evaluates machine learning models for predicting late deliveries.

Using the DataCo SMART Supply Chain dataset, the project combines Python-based data analysis, PostgreSQL business intelligence, statistical exploration, machine learning and an interactive Power BI dashboard.

The objective is to transform raw supply chain data into meaningful business insights that can support delivery monitoring, operational analysis and data-driven decision-making.

## Business Objectives

- Analyse historical supply chain delivery performance.
- Identify delivery patterns across shipping modes, markets, regions and customer segments.
- Investigate order cancellations and their distribution.
- Evaluate financial performance across markets and customer segments.
- Develop machine learning models to identify orders at risk of late delivery.
- Present business insights through an interactive Power BI dashboard.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Database | PostgreSQL 18 |
| SQL Management | pgAdmin 4 |
| Business Intelligence | Microsoft Power BI Desktop |
| Dashboard Calculations | DAX |
| Development Environment | Visual Studio Code |

## Dataset

**DataCo SMART Supply Chain Dataset**

Source: [Mendeley Data – DataCo SMART Supply Chain](https://data.mendeley.com/datasets/8gx2fvg2k6/5)

DOI: https://doi.org/10.17632/8gx2fvg2k6.5

The dataset contains approximately 180,519 order-item records and 53 original attributes.

For this project, the data was transformed into an order-level analytical dataset containing 65,752 unique orders.

The dataset is licensed under Creative Commons Attribution 4.0 (CC BY 4.0).

Attribution:

Constante, F., Silva, F., & Pereira, A. (2019). *DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS* (Version 5). Mendeley Data. https://doi.org/10.17632/8gx2fvg2k6.5.

## Project Workflow

### 1. Data Understanding and Cleaning

- Examined the original dataset's structure, dimensions and data types.
- Investigated missing values, duplicate records and inconsistent order-level attributes.
- Removed irrelevant and personal customer information from the analytical dataset.
- Aggregated order-item records into one row per order.
- Separated cancelled orders from the non-cancelled modelling dataset.

### 2. Exploratory Data Analysis

- Examined delivery delay distributions and late delivery patterns.
- Compared delivery performance across shipping modes and geographical markets.
- Analysed monthly order volumes and delivery trends.
- Investigated customer segment profitability and financial performance.
- Explored relationships between numerical variables and delivery risk.

### 3. Feature Engineering and Machine Learning

Developed two classification models:

- Logistic Regression
- Random Forest Classifier

Applied preprocessing pipelines, categorical encoding, feature scaling and stratified splitting for initial model evaluation.

Subsequently, performed chronological validation to assess model performance on later orders and investigated potential data leakage.

### 4. Model Evaluation and Threshold Optimisation

Evaluated models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Balanced Accuracy
- Confusion Matrices

Optimised classification thresholds using validation data and selected thresholds based on balanced accuracy.

### 5. PostgreSQL Business Analysis

Loaded the analytical dataset into PostgreSQL and performed SQL-based business analysis.

Created seven reusable analytical views covering:

1. Customer Segment Performance
2. Delivery Performance by Market
3. Delivery Performance by Region
4. Delivery Performance by Shipping Mode
5. Executive KPIs
6. Financial Performance by Market
7. Monthly Delivery Performance

### 6. Power BI Dashboard

Developed a three-page interactive dashboard:

**Page 1 – Executive Overview**
- Total orders, sales and profit.
- Late delivery and cancellation rates.
- Monthly order trends.
- Sales and profit by market.
- Delivery status distribution.
- Interactive filters.

**Page 2 – Delivery Risk Analysis**
- Late delivery rates by region and market.
- Average delay by shipping mode.
- Monthly late delivery trends.
- Delivery delay distribution.
- Scheduled versus actual shipping days.

**Page 3 – Predictive Model Performance**
- Model performance comparison.
- ROC-AUC comparison.
- Confusion matrices.
- Random Forest feature importance.
- Precision versus recall comparison.
- Key model findings and limitations.

## Key Business Findings

- The dataset contains 65,752 unique orders.
- The cancellation rate is approximately 4.34%.
- The late delivery rate among non-cancelled orders is approximately 57.31%.
- First Class orders show a 100% late delivery rate in this dataset.
- Second Class orders have an average delay of approximately two days.
- Regional late delivery rates vary, with Central Africa showing approximately 60.04%.
- Europe records the highest total sales and profit among the five markets analysed.

## Machine Learning Results

Chronological test-set results after threshold optimisation:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 69.54% | 69.07% |
| Precision | 88.88% | 86.83% |
| Recall | 53.82% | 54.54% |
| F1 Score | 67.05% | 67.00% |
| ROC-AUC | 74.89% | 73.75% |
| Balanced Accuracy | 72.34% | 71.66% |

Logistic Regression performed slightly better overall across most evaluation metrics, while Random Forest achieved marginally higher recall.

## Limitations and Considerations

- The models were evaluated using a chronological test split from the same dataset; independent external validation is required before real-world deployment.
- The original late delivery risk label was found to match observed positive delivery delays. This confirms label consistency but also means the target is not independent of the recorded delivery outcome.
- Direct post-delivery information was excluded from predictive model inputs to reduce data leakage.
- Shipping mode and scheduled shipping days exhibit a strong relationship with the target in this dataset.
- Feature importance indicates model associations, not causal relationships.
- The dataset represents historical records and may not reflect current supply chain operations.

## Repository Structure

```text
supply-chain-risk-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│
├── sql/
│   ├── schema.sql
│   ├── data_quality_checks.sql
│   └── business_analysis.sql
│
├── powerbi/
│   └── supply_chain_dashboard.pbix
│
├── reports/
│
│
├── requirements.txt
└── README.md
```

## Installation and Reproducibility

Clone the repository:

```bash
git clone https://github.com/RamyaMuthumariappan/supply-chain-risk-intelligence.git
```

Navigate to the project directory:

```bash
cd supply-chain-risk-intelligence
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The raw and processed datasets, PostgreSQL database and Power BI report should be set up separately as described in the project documentation.

## Author

Developed as a portfolio project demonstrating practical skills in data analysis, SQL, machine learning and business intelligence.
