# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

**Understand customers. Forecast demand. Make smarter inventory decisions.**

RetailPulse is an AI-powered retail analytics platform designed to analyze transaction data, understand customer purchasing behavior, forecast future demand, and support inventory planning through an interactive dashboard.

## 🚀 Live Demo

**Try RetailPulse:** https://retailpulse-cup9epsag25pufyrah95ue.streamlit.app/

**GitHub Repository:** https://github.com/HariniSri29-web/RetailPulse

## 📌 Project Overview

Retail businesses generate large volumes of transactional data. Extracting meaningful insights from this data can help businesses understand sales performance, identify valuable customers, estimate churn risk, anticipate demand, and make informed inventory decisions.

RetailPulse brings these analytical workflows together in a centralized dashboard built with Python and Streamlit.

## ✨ Key Features

### 1. Sales Analytics
- Analyze historical sales and revenue trends.
- Explore total revenue, orders, customers, and products.
- Visualize revenue by month, product, and country.
- Filter sales records by date range and download filtered data.

### 2. ETL and Data Quality
- Prepare retail transaction data for analysis.
- Examine missing values and return transactions.
- Identify cancelled invoices and review data quality statistics.
- Display the ETL processing report when available.

### 3. Customer Analytics and Segmentation
- Summarize customer spending and purchase frequency.
- Identify high-value customers.
- Apply Recency, Frequency, and Monetary (RFM) analysis to segment customers.
- Display generated RFM results when available.

### 4. Customer Churn Prediction
- Use machine learning workflows to estimate customer churn risk.
- Review generated churn prediction results when available.
- Explore model feature importance through SHAP explainability.

*Churn estimates should be validated against an appropriate observation window before being used for business decisions.*

### 5. Demand Forecasting
- Analyze historical revenue patterns.
- Support Prophet-based forecasting workflows.
- Explore LSTM-based deep learning forecasts when generated.
- Compare model evaluation metrics, including MAE, RMSE, and MAPE, when available.

### 6. Inventory Optimization
- Generate inventory planning recommendations from historical sales.
- Explore product-level recommendations and filter results by action.
- Download inventory recommendation data when available.

*Recommendations are estimates based on historical demand; actual stock levels and supplier lead times should be considered separately.*

### 7. Model Explainability
- Use SHAP-based feature importance to examine model predictions.
- Display feature importance tables and visualizations when generated.

### 8. Model Monitoring
- Review historical feature distribution changes.
- Examine potential data drift using generated monitoring reports.
- Support investigation of changes that may affect model reliability.

### 9. Interactive Dashboard
- Navigate between business overview, sales analysis, customer insights, forecasting and inventory, data quality, and model monitoring.
- Explore interactive charts, data tables, and downloadable reports.

### 10. Docker Support
- Package the dashboard and its Python dependencies into a Docker image.
- Support consistent local execution through containerization.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Streamlit, Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Demand Forecasting | Prophet |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Model Explainability | SHAP |
| Experiment Tracking | MLflow |
| Data Storage | CSV datasets |
| Containerization | Docker |
| Version Control | Git, GitHub |
| Public Deployment | Streamlit Community Cloud |

## 📂 Project Structure

```text
RetailPulse/
├── app.py
├── prepare_data.py
├── etl_pipeline.py
├── eda.py
├── rfm_segmentation.py
├── churn_prediction.py
├── churn_explainability.py
├── demand_forecasting.py
├── lstm_forecasting.py
├── inventory_optimization.py
├── model_monitoring.py
├── mlflow_tracking.py
├── inspect_data.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── src/
    └── clean_data.py
```

Generated reports, datasets, and model outputs may be stored in additional folders created by the project workflows.

## 📊 Dataset

RetailPulse uses the **Online Retail II** transactional dataset for retail analytics and forecasting.

The dataset contains historical transaction information such as invoice identifiers, product codes, descriptions, quantities, transaction dates, prices, customer identifiers, and countries.

### Data Preparation

1. Obtain the Online Retail II dataset from the original dataset provider.
2. Place the source dataset in the project directory.
3. Follow the input-file requirements specified in `prepare_data.py` and `etl_pipeline.py`.
4. Run the relevant preparation workflow to generate the processed dataset.
5. Launch the dashboard using the generated dataset expected by `app.py`.

**Dataset source:** [Online Retail II — Kaggle](https://www.kaggle.com/datasets/cgrymn/online-retail-ii-uci-dataset)

The dataset contains historical customer and transaction information. Customer identifiers may be missing for some transactions, which should be considered during customer-level analysis.

## ⚙️ Installation and Local Setup

### Prerequisites

- Python 3.12 recommended
- Git
- pip
- The required retail dataset

### 1. Clone the Repository

```bash
git clone https://github.com/HariniSri29-web/RetailPulse.git
cd RetailPulse
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, use the Python executable inside `.venv` directly to install packages and run the application.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Follow the input and output file requirements in the data preparation scripts. Ensure that the processed dataset is available under the filename and location expected by `app.py`.

For the current application, the configured dataset filename is:

```text
retailpulse_prepared.csv.gz
```

The application uses Pandas to load the compressed CSV dataset.

### 5. Run the Dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🐳 Run with Docker

Make sure Docker Desktop is running and the required dataset is available in the Docker build context.

### 1. Build the Docker Image

```powershell
docker build -t retailpulse:1.0 .
```

### 2. Start the Container

```powershell
docker run --rm -p 8501:8501 retailpulse:1.0
```

### 3. Open the Application

Visit:

```text
http://localhost:8501
```

The Docker image must include or otherwise have access to the dataset required by the application.

## 🌐 Deployment

RetailPulse is publicly deployed using **Streamlit Community Cloud**.

**Live application:** https://retailpulse-cup9epsag25pufyrah95ue.streamlit.app/

The source code is maintained in GitHub, and the deployed application is configured from the repository's `main` branch.

The full dataset is stored in compressed CSV format in the repository to support the deployed application.

## 🔮 Future Enhancements

- Add automated model retraining and scheduled forecasting.
- Introduce comprehensive model validation and backtesting.
- Improve monitoring with automated alerts and model performance tracking.
- Integrate real-time retail transaction data.
- Incorporate supplier lead times and actual stock levels into inventory planning.
- Add authentication and role-based dashboard access.
- Optimize performance for larger datasets and concurrent users.

## 🎯 Project Objectives

- Convert retail transaction data into actionable business insights.
- Understand customer purchasing behavior and value.
- Estimate churn risk using machine learning.
- Forecast revenue and demand using time-series and deep learning approaches.
- Support inventory planning with data-driven recommendations.
- Present analytics through an interactive, publicly accessible dashboard.

## 👩‍💻 Author

**HariniSri**  
B.Tech — Artificial Intelligence and Data Science  
Class of 2027

## 📄 License

No license has been specified yet. Add a `LICENSE` file if you intend to distribute the project under an open-source license.

---

**RetailPulse — Understand customers. Forecast demand. Make smarter inventory decisions.**
