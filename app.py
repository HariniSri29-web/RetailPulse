
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="RetailPulse | AI Retail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHART_TEMPLATE = "plotly_white"
NAVY = "#173B5E"
TEAL = "#0D9488"
BLUE = "#3B82F6"


# =========================================================
# 2. PROFESSIONAL UI THEME
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: #F3F6FB;
        color: #172B4D;
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: #142D4E !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown {
        color: #34465D;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102A43, #163B5C);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #F4F8FC !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF, #F0FAFA);
        border: 1px solid #D9E7F0;
        border-left: 4px solid #12B8A6;
        padding: 18px;
        border-radius: 13px;
        box-shadow: 0 4px 14px rgba(20, 45, 78, 0.06);
    }

    div[data-testid="stMetricLabel"] p {
        color: #526579 !important;
        font-size: 14px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #142D4E !important;
        font-weight: 750;
    }

    .rp-banner {
        background: linear-gradient(
            120deg, #102A43 0%, #075E70 58%, #087F8C 100%
        );
        padding: 30px 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(16, 42, 67, 0.14);
    }

    .rp-eyebrow {
        color: #5EEAD4;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 3px;
        margin-bottom: 8px;
    }

    .rp-title {
        color: #FFFFFF !important;
        font-size: 39px;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }

    .rp-subtitle {
        color: #E0F2FE !important;
        font-size: 16px;
        margin-top: 10px;
        margin-bottom: 0;
    }

    .rp-section {
        background: #FFFFFF;
        padding: 16px 20px;
        border: 1px solid #E0E8F0;
        border-radius: 12px;
        margin: 12px 0 18px 0;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: #087F8C;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #066570;
        color: #FFFFFF;
        border: none;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #DCE5EF;
        border-radius: 10px;
    }

    hr {
        border-color: #DCE5EF;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HEADER
# =========================================================
st.markdown("""
<div class="rp-banner">
    <div class="rp-eyebrow">INTELLIGENT RETAIL ANALYTICS</div>
    <div class="rp-title">RetailPulse</div>
    <p class="rp-subtitle">
        Understand customers. Forecast demand.
        Make smarter inventory decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 4. LOAD DATA
# =========================================================
DATA_FILE = BASE_DIR / "retailpulse_prepared.csv.gz"


@st.cache_data
def load_sales_data(path):
    df = pd.read_csv(
        path,
        usecols=[
            "Invoice", "StockCode", "Description", "Quantity",
            "InvoiceDate", "Price", "Customer ID", "Country",
            "TotalAmount", "IsReturn", "IsCancelledInvoice"
        ],
        low_memory=False
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    for column in ["Quantity", "Price", "TotalAmount"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["Customer ID"] = df["Customer ID"].astype("string")

    for column in ["IsCancelledInvoice", "IsReturn"]:
        df[column] = (
            df[column].astype(str).str.lower()
            .isin(["true", "1", "yes"])
        )

    return df


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info(
        "Make sure retailpulse_prepared.csv is inside your RetailPulse folder."
    )
    st.stop()

with st.spinner("Loading RetailPulse analytics..."):
    df = load_sales_data(str(DATA_FILE))

sales = df[
    (df["Quantity"] > 0)
    & (df["Price"] > 0)
    & (~df["IsCancelledInvoice"])
    & df["InvoiceDate"].notna()
].copy()

sales["Revenue"] = sales["Quantity"] * sales["Price"]

if sales.empty:
    st.error("No valid sales records were found in the dataset.")
    st.stop()


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================
st.sidebar.markdown("## RETAILPULSE")
st.sidebar.caption("AI RETAIL INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Sales Analysis",
        "Customer Insights",
        "Forecast & Inventory",
        "Data Quality",
        "Model Monitoring",
        "Product Performance",
        "Regional & Geographic Analytics",
        "Order & Transaction Analytics",
        "Business Intelligence & Insights"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**Project Modules**")
st.sidebar.caption("Sales Analytics")
st.sidebar.caption("Customer Segmentation")
st.sidebar.caption("Churn Prediction")
st.sidebar.caption("Demand Forecasting")
st.sidebar.caption("LSTM Deep Learning")
st.sidebar.caption("Inventory Planning")
st.sidebar.caption("Model Monitoring")
st.sidebar.divider()
st.sidebar.caption("RetailPulse | 2026")


# =========================================================
# 6. OVERVIEW
# =========================================================
if page == "Overview":
    st.header("Business Overview")
    st.caption(
        "A consolidated view of retail business performance."
    )

    total_revenue = sales["Revenue"].sum()
    total_orders = sales["Invoice"].nunique()
    total_customers = sales["Customer ID"].nunique()
    total_products = sales["StockCode"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Revenue", f"{total_revenue:,.2f}")
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Identified Customers", f"{total_customers:,}")
    c4.metric("Unique Products", f"{total_products:,}")

    monthly = (
        sales.assign(
            Month=sales["InvoiceDate"].dt.to_period("M").astype(str)
        )
        .groupby("Month", as_index=False)["Revenue"]
        .sum()
    )

    st.subheader("Revenue Performance")
    st.caption(
        "Track how revenue changes over the historical dataset."
    )

    if not monthly.empty:
        fig = px.line(
            monthly,
            x="Month",
            y="Revenue",
            markers=True,
            template=CHART_TEMPLATE,
            color_discrete_sequence=[TEAL]
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Top 10 Products")

        top_products = (
            sales.groupby("Description", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
            .head(10)
            .sort_values("Revenue")
        )

        fig = px.bar(
            top_products,
            x="Revenue",
            y="Description",
            orientation="h",
            template=CHART_TEMPLATE,
            color="Revenue",
            color_continuous_scale=["#BFEAE5", TEAL]
        )
        fig.update_layout(
            yaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Revenue by Country")

        country_sales = (
            sales.groupby("Country", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
            .head(10)
        )

        fig = px.bar(
            country_sales,
            x="Country",
            y="Revenue",
            template=CHART_TEMPLATE,
            color_discrete_sequence=[BLUE]
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Revenue"
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 7. SALES ANALYSIS
# =========================================================
elif page == "Sales Analysis":
    st.header("Sales Analysis")
    st.caption(
        "Explore transaction-level performance over a selected period."
    )

    min_date = sales["InvoiceDate"].min().date()
    max_date = sales["InvoiceDate"].max().date()

    date_range = st.date_input(
        "Select sales date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    filtered = sales.copy()

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            filtered["InvoiceDate"].dt.date.between(
                start_date, end_date
            )
        ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", f"{filtered['Revenue'].sum():,.2f}")
    c2.metric("Units Sold", f"{filtered['Quantity'].sum():,.0f}")
    c3.metric("Orders", f"{filtered['Invoice'].nunique():,}")

    daily = (
        filtered.assign(Date=filtered["InvoiceDate"].dt.date)
        .groupby("Date", as_index=False)["Revenue"]
        .sum()
    )

    st.subheader("Daily Revenue Trend")

    if not daily.empty:
        fig = px.line(
            daily,
            x="Date",
            y="Revenue",
            template=CHART_TEMPLATE,
            color_discrete_sequence=[TEAL]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sales records exist for this date range.")

    st.subheader("Sales Records")

    st.dataframe(
        filtered[
            [
                "InvoiceDate", "Invoice", "StockCode", "Description",
                "Quantity", "Price", "Revenue", "Country"
            ]
        ].head(1000),
        use_container_width=True
    )

    st.download_button(
        "Download Filtered Sales",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_sales.csv",
        mime="text/csv"
    )


# =========================================================
# 8. CUSTOMER INSIGHTS
# =========================================================
elif page == "Customer Insights":
    st.header("Customer Intelligence")
    st.caption(
        "Understand customer value, purchase behavior, and risk."
    )

    known_customers = sales.dropna(
        subset=["Customer ID"]
    ).copy()

    customer_summary = (
        known_customers.groupby("Customer ID", as_index=False)
        .agg(
            TotalSpent=("Revenue", "sum"),
            TotalOrders=("Invoice", "nunique"),
            UnitsPurchased=("Quantity", "sum"),
            LastPurchase=("InvoiceDate", "max")
        )
        .sort_values("TotalSpent", ascending=False)
    )

    c1, c2 = st.columns(2)
    c1.metric(
        "Identified Customers",
        f"{len(customer_summary):,}"
    )

    c2.metric(
        "Average Customer Spend",
        f"{customer_summary['TotalSpent'].mean():,.2f}"
        if not customer_summary.empty else "0.00"
    )

    st.subheader("Highest-Spending Customers")

    if not customer_summary.empty:
        top_customers = customer_summary.head(10).copy()
        top_customers["Customer ID"] = (
            top_customers["Customer ID"].astype(str)
        )

        fig = px.bar(
            top_customers.sort_values("TotalSpent"),
            x="TotalSpent",
            y="Customer ID",
            orientation="h",
            template=CHART_TEMPLATE,
            color_discrete_sequence=[TEAL]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Summary")
    st.dataframe(
        customer_summary.head(1000),
        use_container_width=True
    )

    rfm_file = (
        BASE_DIR / "rfm_outputs" / "customer_rfm_segments.csv"
    )
    churn_file = (
        BASE_DIR / "churn_outputs" / "customer_churn_predictions.csv"
    )

    if rfm_file.exists():
        st.subheader("RFM Segmentation")
        rfm = pd.read_csv(rfm_file)
        st.caption("Customer segments generated by the RFM model.")
        st.dataframe(rfm.head(100), use_container_width=True)
    else:
        st.info("Run rfm_segmentation.py to generate RFM results.")

    if churn_file.exists():
        st.subheader("Churn Risk Estimates")
        churn = pd.read_csv(churn_file)

        st.warning(
            "These are historical inactivity-risk estimates, not confirmed "
            "churn. Validate the observation window before business use."
        )

        st.dataframe(churn.head(100), use_container_width=True)
    else:
        st.info(
            "Run churn_prediction.py to generate churn results."
        )


# =========================================================
# 9. FORECAST AND INVENTORY
# =========================================================
elif page == "Forecast & Inventory":
    st.header("Forecasting & Inventory Planning")
    st.caption(
        "Explore traditional forecasting, LSTM predictions, "
        "and product planning estimates."
    )

    forecast_file = (
        BASE_DIR / "forecast_outputs" / "daily_revenue_forecast.csv"
    )
    inventory_file = (
        BASE_DIR / "inventory_outputs" / "inventory_recommendations.csv"
    )

    # -----------------------------------------
    # A. EXISTING REVENUE FORECAST
    # -----------------------------------------
    st.subheader("Existing Revenue Forecast")

    if forecast_file.exists():
        forecast = pd.read_csv(forecast_file)

        date_column = (
            "Date" if "Date" in forecast.columns else "ds"
        )

        value_column = (
            "ForecastRevenue"
            if "ForecastRevenue" in forecast.columns else "yhat"
        )

        if date_column in forecast.columns and value_column in forecast.columns:
            forecast[date_column] = pd.to_datetime(
                forecast[date_column], errors="coerce"
            )
            forecast[value_column] = pd.to_numeric(
                forecast[value_column], errors="coerce"
            )

            forecast = forecast.dropna(
                subset=[date_column, value_column]
            )

            if not forecast.empty:
                fig = px.line(
                    forecast,
                    x=date_column,
                    y=value_column,
                    template=CHART_TEMPLATE,
                    color_discrete_sequence=[TEAL]
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Forecast Revenue"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    forecast.tail(30),
                    use_container_width=True
                )

                st.download_button(
                    "Download Existing Forecast",
                    data=forecast.to_csv(index=False).encode("utf-8"),
                    file_name="retailpulse_forecast.csv",
                    mime="text/csv",
                    key="download_existing_forecast"
                )
            else:
                st.warning("The existing forecast contains no valid rows.")
        else:
            st.warning(
                "Expected date or forecast-value columns were not found "
                "in the existing forecast CSV."
            )
    else:
        st.info(
            "Existing forecast unavailable. "
            "Run demand_forecasting.py first."
        )

    # -----------------------------------------
    # B. LSTM DEEP LEARNING FORECAST
    # -----------------------------------------
    st.divider()
    st.subheader("LSTM Deep Learning Forecast")
    st.caption(
        "A recurrent neural network trained on historical daily revenue."
    )

    lstm_dir = BASE_DIR / "lstm_outputs"

    lstm_forecast_file = (
        lstm_dir / "lstm_future_forecast.csv"
    )
    lstm_metrics_file = (
        lstm_dir / "lstm_evaluation.csv"
    )
    lstm_test_file = (
        lstm_dir / "lstm_test_predictions.csv"
    )
    lstm_chart_file = (
        lstm_dir / "lstm_forecast.png"
    )

    # LSTM evaluation metrics
    if lstm_metrics_file.exists():
        lstm_metrics = pd.read_csv(lstm_metrics_file)

        st.markdown("#### Model Evaluation")
        st.dataframe(
            lstm_metrics,
            use_container_width=True
        )

        st.download_button(
            "Download LSTM Evaluation",
            data=lstm_metrics.to_csv(index=False).encode("utf-8"),
            file_name="lstm_evaluation.csv",
            mime="text/csv",
            key="download_lstm_metrics"
        )
    else:
        st.info(
            "LSTM evaluation results are not available yet. "
            "Run lstm_forecasting.py first."
        )

    # LSTM future forecast
    if lstm_forecast_file.exists():
        lstm_forecast = pd.read_csv(lstm_forecast_file)

        required_columns = {
            "Date", "LSTMForecastRevenue"
        }

        if required_columns.issubset(lstm_forecast.columns):
            lstm_forecast["Date"] = pd.to_datetime(
                lstm_forecast["Date"], errors="coerce"
            )

            lstm_forecast["LSTMForecastRevenue"] = pd.to_numeric(
                lstm_forecast["LSTMForecastRevenue"],
                errors="coerce"
            )

            lstm_forecast = lstm_forecast.dropna(
                subset=["Date", "LSTMForecastRevenue"]
            )

            if not lstm_forecast.empty:
                st.markdown("#### Next 30 Days")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Forecast Horizon",
                    f"{len(lstm_forecast)} days"
                )

                c2.metric(
                    "Total Forecast Revenue",
                    f"{lstm_forecast['LSTMForecastRevenue'].sum():,.2f}"
                )

                fig = px.line(
                    lstm_forecast,
                    x="Date",
                    y="LSTMForecastRevenue",
                    markers=True,
                    template=CHART_TEMPLATE,
                    color_discrete_sequence=[BLUE],
                    title="LSTM Daily Revenue Forecast"
                )

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Forecast Revenue",
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    lstm_forecast,
                    use_container_width=True
                )

                st.download_button(
                    "Download LSTM Forecast",
                    data=lstm_forecast.to_csv(index=False).encode("utf-8"),
                    file_name="lstm_future_forecast.csv",
                    mime="text/csv",
                    key="download_lstm_forecast"
                )
        else:
            st.warning(
                "The LSTM forecast file is missing the expected columns: "
                "Date and LSTMForecastRevenue."
            )
    else:
        st.info(
            "LSTM forecast not found. Run lstm_forecasting.py first."
        )

    # Actual versus predicted test-period revenue
    if lstm_test_file.exists():
        test_results = pd.read_csv(lstm_test_file)

        required_test_columns = {
            "Date", "ActualRevenue", "PredictedRevenue"
        }

        if required_test_columns.issubset(test_results.columns):
            test_results["Date"] = pd.to_datetime(
                test_results["Date"], errors="coerce"
            )

            for column in ["ActualRevenue", "PredictedRevenue"]:
                test_results[column] = pd.to_numeric(
                    test_results[column], errors="coerce"
                )

            test_results = test_results.dropna(
                subset=[
                    "Date", "ActualRevenue", "PredictedRevenue"
                ]
            )

            st.markdown("#### Actual vs Predicted Revenue")

            fig = px.line(
                test_results,
                x="Date",
                y=["ActualRevenue", "PredictedRevenue"],
                template=CHART_TEMPLATE,
                title="LSTM Test-Period Performance"
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Revenue",
                legend_title="Series",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.download_button(
                "Download LSTM Test Predictions",
                data=test_results.to_csv(index=False).encode("utf-8"),
                file_name="lstm_test_predictions.csv",
                mime="text/csv",
                key="download_lstm_test"
            )
        else:
            st.warning(
                "The LSTM test file is missing expected columns."
            )

    # Saved LSTM chart
    if lstm_chart_file.exists():
        with st.expander("View Saved LSTM Forecast Chart"):
            st.image(
                str(lstm_chart_file),
                caption="Saved LSTM forecasting visualization",
                use_container_width=True
            )

    st.caption(
        "Forecasts are model estimates, not guaranteed future revenue. "
        "Evaluate models on comparable test periods before choosing one."
    )

    # -----------------------------------------
    # C. INVENTORY RECOMMENDATIONS
    # -----------------------------------------
    st.divider()
    st.subheader("Inventory Recommendations")

    if inventory_file.exists():
        inventory = pd.read_csv(inventory_file)

        st.info(
            "Reorder levels are planning estimates based on historical sales. "
            "Actual stock quantities and supplier lead times are not provided "
            "by the original dataset."
        )

        if "Recommendation" in inventory.columns:
            actions = sorted(
                inventory["Recommendation"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_action = st.selectbox(
                "Filter recommendation",
                ["All"] + actions
            )

            if selected_action != "All":
                inventory = inventory[
                    inventory["Recommendation"] == selected_action
                ]

        st.dataframe(
            inventory.head(1000),
            use_container_width=True
        )

        st.download_button(
            "Download Inventory Recommendations",
            data=inventory.to_csv(index=False).encode("utf-8"),
            file_name="inventory_recommendations.csv",
            mime="text/csv",
            key="download_inventory"
        )
    else:
        st.info(
            "Inventory recommendations unavailable. "
            "Run inventory_optimization.py first."
        )

    # Forecast Model Comparison
    st.subheader("Forecast Model Comparison")
    st.caption(
        "Both models evaluated on the same 30 dates: "
        "10 Nov – 9 Dec 2011."
    )

    comparison_file = (
        BASE_DIR
        / "forecast_outputs"
        / "model_comparison_metrics.csv"
    )

    if comparison_file.exists():
        comparison_df = pd.read_csv(comparison_file)

        st.dataframe(
            comparison_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                comparison_df,
                x="Model",
                y=["MAE", "RMSE"],
                barmode="group",
                title="MAE vs RMSE",
                template=CHART_TEMPLATE,
                labels={
                    "value": "Revenue Error",
                    "variable": "Metric"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                comparison_df,
                x="Model",
                y="MAPE_nonzero_percent",
                title="MAPE Comparison",
                template=CHART_TEMPLATE,
                labels={
                    "MAPE_nonzero_percent": "MAPE (%)",
                    "Model": "Forecasting Model"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "The existing forecasting model has lower MAE, RMSE, "
            "and MAPE on this shared evaluation period. "
            "Performance may differ on other dates."
        )
    else:
        st.warning(
            "Comparison results not found. Run the model "
            "comparison script first."
        )



# =========================================================
# 10. DATA QUALITY AND ETL
# =========================================================
elif page == "Data Quality":
    st.header("Data Quality & ETL")
    st.caption(
        "Review missing values, returns, and data preparation."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Dataset Rows", f"{len(df):,}")

    c2.metric(
        "Missing Customer IDs",
        f"{df['Customer ID'].isna().sum():,}"
    )

    c3.metric(
        "Return Records",
        f"{int(df['IsReturn'].sum()):,}"
    )

    st.subheader("Missing Values by Column")

    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    missing = missing.sort_values(
        "Missing Values", ascending=False
    )

    fig = px.bar(
        missing,
        x="Column",
        y="Missing Values",
        template=CHART_TEMPLATE,
        color_discrete_sequence=[BLUE]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    report_file = (
        BASE_DIR / "etl_outputs" / "etl_report.txt"
    )

    if report_file.exists():
        st.subheader("ETL Processing Report")
        st.code(
            report_file.read_text(encoding="utf-8"),
            language="text"
        )
    else:
        st.info(
            "Run etl_pipeline.py to generate the ETL report."
        )

    st.subheader("Dataset Preview")
    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# =========================================================
# 11. MODEL MONITORING AND EXPLAINABLE AI
# =========================================================
elif page == "Model Monitoring":
    st.header("Model Monitoring & Explainable AI")
    st.caption(
        "Review churn-model feature importance and historical "
        "changes in customer data."
    )

    # -----------------------------------------
    # A. SHAP FEATURE IMPORTANCE
    # -----------------------------------------
    shap_file = (
        BASE_DIR
        / "shap_outputs"
        / "shap_feature_importance.csv"
    )

    st.subheader("Churn Model Explainability (SHAP)")

    if shap_file.exists():
        shap_df = pd.read_csv(shap_file)

        st.caption(
            "SHAP estimates how input features contribute to model "
            "predictions. Global feature importance indicates overall "
            "influence; it does not by itself show the direction of impact."
        )

        st.dataframe(
            shap_df,
            use_container_width=True
        )

        feature_candidates = [
            "Feature", "feature", "Feature Name", "feature_name"
        ]

        
        importance_candidates = [
            "MeanAbsoluteSHAP",
            "MeanAbsSHAP",
            "mean_abs_shap",
            "Importance",
            "importance",
            "Mean SHAP",
            "mean_shap"
        ]

        feature_col = next(
            (
                column for column in feature_candidates
                if column in shap_df.columns
            ),
            None
        )

        importance_col = next(
            (
                column for column in importance_candidates
                if column in shap_df.columns
            ),
            None
        )

        if feature_col and importance_col:
            plot_df = shap_df.sort_values(
                importance_col,
                ascending=True
            )

            fig = px.bar(
                plot_df,
                x=importance_col,
                y=feature_col,
                orientation="h",
                template=CHART_TEMPLATE,
                title="Global SHAP Feature Importance",
                color_discrete_sequence=[TEAL]
            )

            fig.update_layout(
                xaxis_title="Mean Absolute SHAP Value",
                yaxis_title="Feature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            st.warning(
                "Could not identify the feature and importance columns "
                "for the SHAP chart. The data table is still available."
            )

        shap_image = (
            BASE_DIR / "shap_outputs" / "shap_summary.png"
        )

        if shap_image.exists():
            st.subheader("SHAP Summary Plot")
            st.image(
                str(shap_image),
                caption="Feature effects across the explained observations",
                use_container_width=True
            )

        st.download_button(
            "Download SHAP Feature Importance",
            data=shap_df.to_csv(index=False).encode("utf-8"),
            file_name="shap_feature_importance.csv",
            mime="text/csv",
            key="download_shap"
        )
    else:
        st.warning(
            "SHAP results were not found. "
            "Run churn_explainability.py first."
        )

    st.divider()

    # -----------------------------------------
    # B. HISTORICAL FEATURE-DRIFT MONITORING
    # -----------------------------------------
    st.subheader("Historical Feature-Drift Monitoring")

    drift_file = (
        BASE_DIR / "monitoring_outputs" / "drift_report.csv"
    )

    if drift_file.exists():
        drift_df = pd.read_csv(drift_file)

        st.info(
            "This report compares historical customer-feature distributions. "
            "Detected drift does not automatically mean model accuracy has "
            "decreased. Investigate the cause before retraining."
        )

        st.dataframe(
            drift_df,
            use_container_width=True
        )

        status_col = next(
            (
                column for column in ["Status", "status"]
                if column in drift_df.columns
            ),
            None
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Features Monitored",
            f"{len(drift_df):,}"
        )

        if status_col:
            status_values = (
                drift_df[status_col].fillna("Unknown").astype(str)
            )

            detected_count = status_values.str.contains(
                r"moderate|high",
                case=False,
                na=False
            ).sum()

            c2.metric(
                "Features with Moderate/High Drift",
                f"{detected_count:,}"
            )

            status_counts = (
                status_values.value_counts()
                .rename_axis("Status")
                .reset_index(name="Feature Count")
            )

            fig = px.bar(
                status_counts,
                x="Status",
                y="Feature Count",
                title="Feature Drift Status Summary",
                template=CHART_TEMPLATE,
                color_discrete_sequence=[BLUE]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
        else:
            c2.metric("Drift Status", "Not available")
            st.warning(
                "No Status column was found in the drift report."
            )

        st.download_button(
            "Download Drift Report",
            data=drift_df.to_csv(index=False).encode("utf-8"),
            file_name="drift_report.csv",
            mime="text/csv",
            key="download_drift"
        )
    else:
        st.warning(
            "Drift report not found. Run model_monitoring.py first."
        )
        
# =========================================================
# 12. PRODUCT PERFORMANCE
# =========================================================
elif page == "Product Performance":
    st.header("Product Performance Analytics")
    st.caption(
        "Analyze product revenue, sales volume, and product contribution."
    )

    # Product-level summary
    product_data = (
        sales.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Units_Sold=("Quantity", "sum"),
            Orders=("Invoice", "nunique")
        )
    )

    product_data = product_data.sort_values(
        "Revenue", ascending=False
    )

    # KPI calculations
    total_products = product_data["StockCode"].nunique()
    total_revenue = product_data["Revenue"].sum()
    total_units = product_data["Units_Sold"].sum()
    average_revenue = (
        product_data["Revenue"].mean()
        if not product_data.empty else 0
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Unique Products", f"{total_products:,}")
    c2.metric("Product Revenue", f"{total_revenue:,.2f}")
    c3.metric("Units Sold", f"{total_units:,.0f}")
    c4.metric("Average Revenue per Product", f"{average_revenue:,.2f}")

    st.divider()

    # Product search
    st.subheader("Product Search")

    search = st.text_input(
        "Search by product name or product code"
    )

    filtered_products = product_data.copy()

    if search:
        filtered_products = filtered_products[
            filtered_products["Description"].astype(str).str.contains(
                search, case=False, na=False
            )
            |
            filtered_products["StockCode"].astype(str).str.contains(
                search, case=False, na=False
            )
        ]

    # Top 10 products by revenue and units
    left, right = st.columns(2)

    with left:
        st.subheader("Top 10 Products by Revenue")

        top_revenue = (
            product_data.nlargest(10, "Revenue")
            .sort_values("Revenue")
        )

        fig = px.bar(
            top_revenue,
            x="Revenue",
            y="Description",
            orientation="h",
            template=CHART_TEMPLATE,
            color="Revenue",
            color_continuous_scale=["#BFEAE5", TEAL]
        )

        fig.update_layout(
            xaxis_title="Revenue",
            yaxis_title="",
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Top 10 Products by Units Sold")

        top_units = (
            product_data.nlargest(10, "Units_Sold")
            .sort_values("Units_Sold")
        )

        fig = px.bar(
            top_units,
            x="Units_Sold",
            y="Description",
            orientation="h",
            template=CHART_TEMPLATE,
            color_discrete_sequence=[BLUE]
        )

        fig.update_layout(
            xaxis_title="Units Sold",
            yaxis_title=""
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Revenue contribution
    st.subheader("Product Revenue Contribution")

    if total_revenue > 0:
        contribution = product_data.nlargest(10, "Revenue").copy()
        contribution["Contribution (%)"] = (
            contribution["Revenue"] / total_revenue * 100
        )

        fig = px.pie(
            contribution,
            names="Description",
            values="Revenue",
            hole=0.48,
            template=CHART_TEMPLATE,
            title="Top 10 Products Revenue Share"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Product details
    st.subheader("Product Performance Details")

    st.dataframe(
        filtered_products,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Product Performance Report",
        data=filtered_products.to_csv(index=False).encode("utf-8"),
        file_name="product_performance_report.csv",
        mime="text/csv"
    )

    # Business insights
    st.divider()
    st.subheader("Business Insights")

    if not product_data.empty:
        best_product = product_data.iloc[0]
        highest_volume = product_data.loc[
            product_data["Units_Sold"].idxmax()
        ]

        st.success(
            f"Top revenue-generating product: "
            f"{best_product['Description']} "
            f"with revenue of {best_product['Revenue']:,.2f}."
        )

        st.info(
            f"Highest volume product: "
            f"{highest_volume['Description']} "
            f"with {highest_volume['Units_Sold']:,.0f} units sold."
        )

        st.write(
            "Recommendation: Prioritize stock planning for products "
            "with consistently high sales volume. Review low-performing "
            "products before making purchasing decisions."
        )
        
elif page == "Regional & Geographic Analytics":

    st.title("🌍 Regional & Geographic Analytics")
    st.markdown("Analyze sales performance and customer distribution across countries.")

    # Country-wise aggregation
    country_data = sales.groupby("Country").agg(
        Total_Revenue=("Revenue", "sum"),
        Total_Orders=("Invoice", "nunique"),
        Total_Units=("Quantity", "sum"),
        Total_Customers=("Customer ID", "nunique")
    ).reset_index()

    country_data = country_data.sort_values(
        "Total_Revenue", ascending=False
    )

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌍 Countries", country_data["Country"].nunique())
    col2.metric("💰 Total Revenue", f"{country_data['Total_Revenue'].sum():,.2f}")
    col3.metric("📦 Total Orders", f"{country_data['Total_Orders'].sum():,}")
    col4.metric("👥 Customers", f"{country_data['Total_Customers'].sum():,}")

    st.divider()

    # Top 10 countries by revenue
    st.subheader("🏆 Top 10 Countries by Revenue")

    top_countries = country_data.head(10)

    st.bar_chart(
        top_countries.set_index("Country")["Total_Revenue"]
    )

    st.divider()

    # Country-wise order analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Orders by Country")
        st.bar_chart(
            top_countries.set_index("Country")["Total_Orders"]
        )

    with col2:
        st.subheader("👥 Customers by Country")
        st.bar_chart(
            top_countries.set_index("Country")["Total_Customers"]
        )

    st.divider()

    # Country search
    st.subheader("🔎 Explore Country Performance")

    selected_country = st.selectbox(
        "Select a country",
        sorted(country_data["Country"].dropna().unique())
    )

    selected_data = country_data[
        country_data["Country"] == selected_country
    ]

    if not selected_data.empty:
        row = selected_data.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Revenue", f"{row['Total_Revenue']:,.2f}")
        col2.metric("Orders", f"{row['Total_Orders']:,}")
        col3.metric("Units Sold", f"{row['Total_Units']:,}")
        col4.metric("Customers", f"{row['Total_Customers']:,}")

    st.divider()

    # Complete country report
    st.subheader("📋 Complete Country Performance Report")

    st.dataframe(
        country_data.style.format({
            "Total_Revenue": "{:,.2f}",
            "Total_Orders": "{:,.0f}",
            "Total_Units": "{:,.0f}",
            "Total_Customers": "{:,.0f}"
        }),
        use_container_width=True
    )

    # Business insight
    best_country = country_data.iloc[0]

    st.success(
        f"💡 {best_country['Country']} is the highest-revenue country "
        f"with revenue of {best_country['Total_Revenue']:,.2f}."
    )
    
elif page == "Order & Transaction Analytics":

    st.title("🧾 Order & Transaction Analytics")
    st.markdown("Explore order patterns, transaction values, and cancellation trends.")

    # Prepare order-level data
    order_data = sales.groupby("Invoice").agg(
        Order_Value=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Order_Date=("InvoiceDate", "min"),
        Customer_ID=("Customer ID", "first"),
        Country=("Country", "first")
    ).reset_index()

    order_data["Order_Date"] = pd.to_datetime(order_data["Order_Date"])
    order_data["YearMonth"] = order_data["Order_Date"].dt.to_period("M").astype(str)

    # KPI metrics
    total_orders = order_data["Invoice"].nunique()
    total_revenue = order_data["Order_Value"].sum()
    avg_order_value = (
        total_revenue / total_orders if total_orders else 0
    )
    avg_quantity = (
        order_data["Total_Quantity"].mean() if total_orders else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🧾 Total Orders", f"{total_orders:,}")
    col2.metric("💰 Total Revenue", f"{total_revenue:,.2f}")
    col3.metric("🛒 Average Order Value", f"{avg_order_value:,.2f}")
    col4.metric("📦 Avg. Order Quantity", f"{avg_quantity:,.1f}")

    st.divider()

    # Monthly order trend
    st.subheader("📈 Monthly Order Trend")

    monthly_orders = (
        order_data.groupby("YearMonth")
        .size()
        .reset_index(name="Total_Orders")
        .sort_values("YearMonth")
    )

    st.line_chart(
        monthly_orders.set_index("YearMonth")["Total_Orders"]
    )

    st.divider()

    # Order value distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Order Value Distribution")
        st.bar_chart(
            order_data.set_index("Invoice")["Order_Value"].head(20)
        )

    with col2:
        st.subheader("📦 Order Quantity Distribution")
        st.bar_chart(
            order_data.set_index("Invoice")["Total_Quantity"].head(20)
        )

    st.divider()

    # Top 10 highest-value orders
    st.subheader("🏆 Top 10 Highest-Value Orders")

    top_orders = order_data.nlargest(10, "Order_Value")

    st.dataframe(
        top_orders,
        use_container_width=True
    )

    st.divider()

    # Country-wise order analysis
    st.subheader("🌍 Orders by Country")

    country_orders = (
        order_data.groupby("Country")
        .size()
        .reset_index(name="Total_Orders")
        .sort_values("Total_Orders", ascending=False)
        .head(10)
    )

    st.bar_chart(
        country_orders.set_index("Country")["Total_Orders"]
    )

    st.divider()

    # Cancellation analysis from original dataset
    st.subheader("🚫 Cancellation Analysis")

    cancelled_invoices = (
        df.loc[
            df["IsCancelledInvoice"],
            "Invoice"
        ].nunique()
    )

    valid_invoices = (
        df.loc[
            ~df["IsCancelledInvoice"],
            "Invoice"
        ].nunique()
    )

    col1, col2 = st.columns(2)

    col1.metric("Cancelled Invoices", f"{cancelled_invoices:,}")
    col2.metric("Non-Cancelled Invoices", f"{valid_invoices:,}")

    st.caption(
        "Cancellation counts are based on the original dataset. "
        "An invoice is counted once even if it contains multiple transaction rows."
    )

    st.divider()

    # Transaction details
    st.subheader("📋 Order Details")

    search_order = st.text_input(
        "Search by Invoice Number"
    )

    filtered_orders = order_data.copy()

    if search_order:
        filtered_orders = filtered_orders[
            filtered_orders["Invoice"].astype(str).str.contains(
                search_order, case=False, na=False
            )
        ]

    st.dataframe(
        filtered_orders.sort_values(
            "Order_Date", ascending=False
        ),
        use_container_width=True
    )
    
elif page == "Business Intelligence & Insights":

    st.title("💡 Business Intelligence & Insights")
    st.markdown(
        "An executive overview of business performance, "
        "growth opportunities, and actionable recommendations."
    )

    # Prepare business metrics
    total_revenue = sales["Revenue"].sum()
    total_orders = sales["Invoice"].nunique()
    total_customers = sales["Customer ID"].nunique()
    total_units = sales["Quantity"].sum()

    avg_order_value = (
        total_revenue / total_orders if total_orders else 0
    )

    # Executive KPI dashboard
    st.subheader("📊 Executive Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Revenue", f"{total_revenue:,.2f}")
    col2.metric("🧾 Total Orders", f"{total_orders:,}")
    col3.metric("👥 Total Customers", f"{total_customers:,}")
    col4.metric("📦 Units Sold", f"{total_units:,}")

    st.divider()

    # Monthly revenue analysis
    st.subheader("📈 Monthly Revenue Performance")

    monthly_revenue = (
        sales.groupby(sales["InvoiceDate"].dt.to_period("M"))["Revenue"]
        .sum()
        .reset_index()
    )

    monthly_revenue.columns = ["Month", "Revenue"]
    monthly_revenue["Month"] = monthly_revenue["Month"].astype(str)
    monthly_revenue = monthly_revenue.sort_values("Month")

    st.line_chart(
        monthly_revenue.set_index("Month")["Revenue"]
    )

    # Monthly growth calculation
    monthly_revenue["Growth_Percentage"] = (
        monthly_revenue["Revenue"].pct_change() * 100
    )

    st.subheader("📉 Month-on-Month Revenue Growth")

    st.dataframe(
        monthly_revenue.tail(12).style.format({
            "Revenue": "{:,.2f}",
            "Growth_Percentage": "{:.2f}%"
        }),
        use_container_width=True
    )

    st.divider()

    # Best-performing product
    st.subheader("🏆 Business Leaders")

    product_performance = (
        sales.groupby(["StockCode", "Description"])
        .agg(
            Revenue=("Revenue", "sum"),
            Units_Sold=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    country_performance = (
        sales.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        if not product_performance.empty:
            best_product = product_performance.iloc[0]
            st.metric(
                "🥇 Best Product",
                str(best_product["Description"])[:35],
                f"Revenue: {best_product['Revenue']:,.2f}"
            )

    with col2:
        if not country_performance.empty:
            best_country = country_performance.index[0]
            st.metric(
                "🌍 Top Revenue Country",
                str(best_country),
                f"Revenue: {country_performance.iloc[0]:,.2f}"
            )

    st.divider()

    # Revenue contribution
    st.subheader("📊 Revenue Contribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 5 Products**")
        st.bar_chart(
            product_performance.head(5).set_index(
                "Description"
            )["Revenue"]
        )

    with col2:
        st.markdown("**Top 5 Countries**")
        st.bar_chart(
            country_performance.head(5)
        )

    st.divider()

    # Business recommendations
    st.subheader("🧠 Business Recommendations")

    if not monthly_revenue.empty:
        if len(monthly_revenue) >= 2:
            latest_growth = monthly_revenue[
                "Growth_Percentage"
            ].iloc[-1]

            if pd.notna(latest_growth):
                if latest_growth > 0:
                    st.success(
                        f"📈 Revenue increased by {latest_growth:.2f}% "
                        "compared with the previous month. "
                        "Continue monitoring the successful sales strategies."
                    )
                elif latest_growth < 0:
                    st.warning(
                        f"⚠️ Revenue decreased by {abs(latest_growth):.2f}% "
                        "compared with the previous month. "
                        "Review pricing, demand, and marketing activities."
                    )
                else:
                    st.info(
                        "Revenue remained unchanged compared with the previous month."
                    )

    if not product_performance.empty:
        st.info(
            f"💡 Focus on the best-performing product: "
            f"{product_performance.iloc[0]['Description']}. "
            "Maintain sufficient inventory to support demand."
        )

    if not country_performance.empty:
        st.info(
            f"🌍 {country_performance.index[0]} generates the highest revenue. "
            "Study its customer and sales patterns to identify opportunities "
            "in other markets."
        )

    st.divider()

    # Executive summary
    st.subheader("📋 Executive Business Summary")

    st.write(
        f"""
        - Total business revenue: **{total_revenue:,.2f}**
        - Total orders processed: **{total_orders:,}**
        - Unique customers: **{total_customers:,}**
        - Total units sold: **{total_units:,}**
        - Average order value: **{avg_order_value:,.2f}**
        """
    )

    st.caption(
        "Business recommendations are generated using rule-based "
        "analysis of the available sales data."
    )






# =========================================================
# 12. FOOTER
# =========================================================
st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:12px 0;">
        <span style="color:#087F8C; font-weight:700;">RetailPulse</span>
        <span style="color:#64748B;">
            &nbsp;| AI-Powered Retail Analytics
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
