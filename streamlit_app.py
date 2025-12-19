import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Copra Production & Price Forecast", layout="wide")

def perform_adf_test(series):
    try:
        result = adfuller(series.dropna())
        return {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Stationary': result[1] < 0.05
        }
    except:
        return {'Test Statistic': 0, 'p-value': 1.0, 'Stationary': False}

def get_arima_forecast(series, steps, order=(1,1,1)):
    try:
        series = series.dropna()
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        
        # Calculate MAPE on existing data
        train_size = int(len(series) * 0.8)
        mape = 0
        if train_size > 5:
            train, test = series[0:train_size], series[train_size:]
            model_eval = ARIMA(train, order=order).fit()
            predictions = model_eval.forecast(steps=len(test))
            mape = mean_absolute_percentage_error(test, predictions)
            
        return forecast, mape, model_fit
    except Exception as e:
        return None, None, None

def main():
    st.title("🥥 Copra Analytics: Production & Price Forecasting")
    st.markdown("Dual ARIMA modeling for Barangay-level production (MT) and Farmgate Prices (PHP/kg) up to 2035.")

    # Sidebar for Upload and Data Management
    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if 'data' not in st.session_state:
        st.session_state.data = None

    if uploaded_file is not None:
        if st.session_state.data is None:
            df = pd.read_csv(uploaded_file)
            df['Period'] = pd.to_datetime(df['Period'])
            st.session_state.data = df
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Data Update/Addition Section
        with st.sidebar.expander("Add/Update Data"):
            with st.form("data_form"):
                new_brgy = st.selectbox("Barangay", df['Barangay'].unique())
                new_year = st.number_input("Year", min_value=2015, max_value=2050, value=2025)
                new_q = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
                new_prod = st.number_input("Production (MT)", min_value=0.0)
                new_price = st.number_input("Farmgate Price (PHP/kg)", min_value=0.0)
                submit = st.form_submit_button("Update Records")
                
                if submit:
                    period_str = f"{new_year}-{'01' if new_q=='Q1' else '04' if new_q=='Q2' else '07' if new_q=='Q3' else '10'}-01"
                    new_row = {
                        'Barangay': new_brgy, 'Year': new_year, 'Quarter': new_q,
                        'Period': pd.to_datetime(period_str),
                        'Copra_Production (MT)': new_prod,
                        'Farmgate Price (PHP/kg)': new_price
                    }
                    mask = (df['Barangay'] == new_brgy) & (df['Year'] == new_year) & (df['Quarter'] == new_q)
                    if mask.any():
                        df.loc[mask, 'Copra_Production (MT)'] = new_prod
                        df.loc[mask, 'Farmgate Price (PHP/kg)'] = new_price
                    else:
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    st.session_state.data = df.sort_values(['Barangay', 'Period'])
                    st.success("Data Updated!")

        # Main Analysis Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Forecasting & Comparison", "📊 Diagnostics & MAPE", "📋 Forecast Table"])

        # Select Barangay
        barangay_list = df['Barangay'].unique()
        selected_brgy = st.selectbox("Select Barangay for Analysis", barangay_list)
        
        brgy_df = df[df['Barangay'] == selected_brgy].sort_values('Period')
        brgy_df.set_index('Period', inplace=True)
        
        prod_series = brgy_df['Copra_Production (MT)']
        price_series = brgy_df['Farmgate Price (PHP/kg)']

        # Forecast Parameters
        with st.sidebar.expander("ARIMA Parameters", expanded=True):
            p = st.slider("AR (p)", 0, 5, 1)
            d = st.slider("I (d)", 0, 2, 1)
            q = st.slider("MA (q)", 0, 5, 1)
            
            last_date = prod_series.index.max()
            target_year = 2035
            months_diff = (target_year - last_date.year) * 12 + (12 - last_date.month)
            steps = int(months_diff / 3) + 4

        # Calculate Forecasts
        prod_fore, prod_mape, _ = get_arima_forecast(prod_series, steps, (p,d,q))
        price_fore, price_mape, _ = get_arima_forecast(price_series, steps, (p,d,q))

        future_dates = pd.date_range(start=prod_series.index[-1] + pd.DateOffset(months=3), periods=steps, freq='QS')

        with tab1:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Production Forecast (MT)")
                fig_prod = go.Figure()
                fig_prod.add_trace(go.Scatter(x=prod_series.index, y=prod_series, name="Actual", line=dict(color='#1f77b4')))
                fig_prod.add_trace(go.Scatter(x=future_dates, y=prod_fore, name="Forecast", line=dict(color='#ff7f0e', dash='dash')))
                fig_prod.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=40))
                st.plotly_chart(fig_prod, use_container_width=True)

            with col_b:
                st.subheader("Farmgate Price Forecast (PHP)")
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=price_series.index, y=price_series, name="Actual", line=dict(color='#2ca02c')))
                fig_price.add_trace(go.Scatter(x=future_dates, y=price_fore, name="Forecast", line=dict(color='#d62728', dash='dash')))
                fig_price.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=40))
                st.plotly_chart(fig_price, use_container_width=True)

            # Bar Chart Comparison
            st.subheader("Yearly Comparison (Actual vs Predicted)")
            hist_yearly = prod_series.resample('YE').mean().reset_index()
            hist_yearly['Type'] = 'Actual'
            fore_yearly = pd.Series(prod_fore.values, index=future_dates).resample('YE').mean().reset_index()
            fore_yearly.columns = ['Period', 'Copra_Production (MT)']
            fore_yearly['Type'] = 'Forecast'
            
            comp_df = pd.concat([hist_yearly, fore_yearly])
            comp_df['Year'] = comp_df['Period'].dt.year
            fig_bar = px.bar(comp_df, x='Year', y='Copra_Production (MT)', color='Type', barmode='group', height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.subheader("Accuracy & Stationarity Metrics")
            m1, m2 = st.columns(2)
            
            # MAPE Section
            with m1:
                st.info("🎯 **Production Model Accuracy**")
                st.metric("Production MAPE", f"{prod_mape:.2%}" if prod_mape else "N/A")
                adf_p = perform_adf_test(prod_series)
                st.write(f"Stationarity (ADF p-value): `{adf_p['p-value']:.4f}`")
                
            with m2:
                st.info("💰 **Price Model Accuracy**")
                st.metric("Price MAPE", f"{price_mape:.2%}" if price_mape else "N/A")
                adf_pr = perform_adf_test(price_series)
                st.write(f"Stationarity (ADF p-value): `{adf_pr['p-value']:.4f}`")

            st.divider()
            
            st.subheader("ACF & PACF Diagnostics")
            diag_choice = st.radio("Show diagnostics for:", ["Production", "Farmgate Price"], horizontal=True)
            diag_series = prod_series if diag_choice == "Production" else price_series
            
            col_acf, col_pacf = st.columns(2)
            with col_acf:
                lag_acf = acf(diag_series.dropna(), nlags=10)
                fig_acf, ax_acf = plt.subplots(figsize=(6, 3))
                ax_acf.stem(lag_acf)
                ax_acf.set_title(f"{diag_choice} ACF")
                st.pyplot(fig_acf)
            with col_pacf:
                lag_pacf = pacf(diag_series.dropna(), nlags=10)
                fig_pacf, ax_pacf = plt.subplots(figsize=(6, 3))
                ax_pacf.stem(lag_pacf)
                ax_pacf.set_title(f"{diag_choice} PACF")
                st.pyplot(fig_pacf)

        with tab3:
            st.subheader(f"Forecasted Data Table for {selected_brgy} (2025-2035)")
            forecast_data = pd.DataFrame({
                'Year': future_dates.year,
                'Quarter': future_dates.quarter.map(lambda x: f"Q{x}"),
                'Predicted_Production_MT': prod_fore.values,
                'Predicted_Farmgate_Price_PHP': price_fore.values
            })
            
            st.dataframe(forecast_data.style.format({
                'Predicted_Production_MT': '{:.2f}',
                'Predicted_Farmgate_Price_PHP': '{:.2f}'
            }), use_container_width=True)
            
            csv = forecast_data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast Report", csv, f"forecast_{selected_brgy}_2035.csv", "text/csv")

    else:
        st.info("Please upload the Copra Production CSV file to begin.")

if __name__ == "__main__":
    main()
