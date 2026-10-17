# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

RetailPulse is a data science and analytics platform designed to analyze retail transactions, understand customer purchasing behavior, forecast future demand, and support inventory planning.

## Key Features

- **Sales Analytics:** Explore retail sales trends and transaction patterns.
- **ETL & Data Quality:** Clean, prepare, and validate retail transaction data.
- **Customer Segmentation:** Apply RFM analysis to understand customer behavior.
- **Churn Prediction:** Estimate customer churn risk using machine learning.
- **Demand Forecasting:** Compare forecasting approaches, including Prophet-based forecasting and LSTM.
- **Inventory Optimization:** Generate insights to support inventory planning.
- **Model Explainability:** Use SHAP to understand feature importance.
- **Model Monitoring:** Examine feature distribution changes and potential data drift.
- **Interactive Dashboard:** Explore insights through a Streamlit application.
- **Containerization:** Package the application with Docker.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Forecasting:** Prophet, TensorFlow/Keras LSTM
- **Visualization:** Streamlit, Plotly, Matplotlib
- **Explainability:** SHAP
- **Experiment Tracking:** MLflow
- **Deployment Packaging:** Docker

## Project Structure

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

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/HariniSri29-web/RetailPulse.git
cd RetailPulse
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the required Online Retail II dataset in the project directory and follow the data preparation instructions in the Python scripts.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Run with Docker

Build the image:

```bash
docker build -t retailpulse:1.0 .
```

Run the container:

```bash
docker run --rm -p 8501:8501 retailpulse:1.0
```

Open `http://localhost:8501` in your browser.

## Project Status

The application has been run locally using Docker. Public deployment, comprehensive model validation, and production monitoring should be treated as separate steps until verified.

## Author

**HariniSri**  
B.Tech — Artificial Intelligence and Data Science
