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
st.set_page_config(page_title="Copra Production ARIMA Forecast", layout="wide")

def perform_adf_test(series):
    result = adfuller(series.dropna())
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4]
    }

def get_arima_forecast(series, steps, order=(1,1,1)):
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        
        # Calculate MAPE on existing data
        train_size = int(len(series) * 0.8)
        if train_size > 5:
            train, test = series[0:train_size], series[train_size:]
            model_eval = ARIMA(train, order=order).fit()
            predictions = model_eval.forecast(steps=len(test))
            mape = mean_absolute_percentage_error(test, predictions)
        else:
            mape = 0
            
        return forecast, mape, model_fit
    except:
        return None, None, None

def main():
    st.title("🥥 Copra Production ARIMA Forecasting System")
    st.markdown("Upload, analyze, and forecast Copra production trends per Barangay up to 2035.")

    # Sidebar for Upload and Data Management
    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    # Initialize session state for data
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
                new_price = st.number_input("Farmgate Price", min_value=0.0)
                submit = st.form_submit_button("Update Records")
                
                if submit:
                    period_str = f"{new_year}-{'01' if new_q=='Q1' else '04' if new_q=='Q2' else '07' if new_q=='Q3' else '10'}-01"
                    new_row = {
                        'Barangay': new_brgy,
                        'Year': new_year,
                        'Quarter': new_q,
                        'Period': pd.to_datetime(period_str),
                        'Copra_Production (MT)': new_prod,
                        'Farmgate Price (PHP/kg)': new_price
                    }
                    # Update if exists, else append
                    mask = (df['Barangay'] == new_brgy) & (df['Year'] == new_year) & (df['Quarter'] == new_q)
                    if mask.any():
                        df.loc[mask, 'Copra_Production (MT)'] = new_prod
                        df.loc[mask, 'Farmgate Price (PHP/kg)'] = new_price
                    else:
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    st.session_state.data = df.sort_values(['Barangay', 'Period'])
                    st.success("Data Updated!")

        # Main Analysis Tabs
        tab1, tab2, tab3 = st.tabs(["📈 Forecasting", "📊 Statistical Analysis", "📋 Data Table"])

        # Select Barangay
        barangay_list = df['Barangay'].unique()
        selected_brgy = st.selectbox("Select Barangay for Analysis", barangay_list)
        
        brgy_df = df[df['Barangay'] == selected_brgy].sort_values('Period')
        brgy_df.set_index('Period', inplace=True)
        series = brgy_df['Copra_Production (MT)']

        with tab1:
            st.subheader(f"ARIMA Forecast for {selected_brgy}")
            
            # Forecast Settings
            col1, col2 = st.columns([1, 3])
            with col1:
                p = st.number_input("AR (p)", 0, 5, 1)
                d = st.number_input("I (d)", 0, 2, 1)
                q = st.number_input("MA (q)", 0, 5, 1)
                
                last_date = series.index.max()
                target_year = 2035
                # Calculate quarters needed
                months_diff = (target_year - last_date.year) * 12 + (12 - last_date.month)
                steps = int(months_diff / 3) + 4
            
            forecast, mape, model_fit = get_arima_forecast(series, steps, order=(p, d, q))
            
            if forecast is not None:
                # Create future dates index
                future_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=3), periods=steps, freq='QS')
                forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)
                
                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index, y=series, name="Historical Data", line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], name="Forecasted", line=dict(color='red', dash='dash')))
                
                fig.update_layout(title=f"Copra Production Forecast (MT) for {selected_brgy}", xaxis_title="Year", yaxis_title="Production (MT)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison Bar Graph (Yearly Average)
                st.subheader("Comparison: Actual vs Forecasted Yearly Average")
                hist_yearly = series.resample('YE').mean().reset_index()
                hist_yearly['Type'] = 'Actual'
                fore_yearly = forecast_df.resample('YE').mean().reset_index()
                fore_yearly.columns = ['Period', 'Copra_Production (MT)']
                fore_yearly['Type'] = 'Forecast'
                
                comp_df = pd.concat([hist_yearly, fore_yearly])
                comp_df['Year'] = comp_df['Period'].dt.year
                
                fig_bar = px.bar(comp_df, x='Year', y='Copra_Production (MT)', color='Type', barmode='group',
                                 title="Yearly Production Comparison")
                st.plotly_chart(fig_bar, use_container_width=True)

                # Display MAPE
                st.metric("Model MAPE (Accuracy Metric)", f"{mape:.2%}")

        with tab2:
            st.subheader("Time Series Diagnostics")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.write("**Augmented Dickey-Fuller (ADF) Test**")
                adf_res = perform_adf_test(series)
                st.write(f"Test Statistic: `{adf_res['Test Statistic']:.4f}`")
                st.write(f"p-value: `{adf_res['p-value']:.4f}`")
                if adf_res['p-value'] < 0.05:
                    st.success("Stationary (p < 0.05)")
                else:
                    st.warning("Non-Stationary - Consider increasing 'd' parameter")
            
            with c2:
                st.write("**ACF & PACF Plots**")
                lag_acf = acf(series.dropna(), nlags=10)
                lag_pacf = pacf(series.dropna(), nlags=10)
                
                fig_corr, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.stem(lag_acf)
                ax1.set_title("ACF")
                ax2.stem(lag_pacf)
                ax2.set_title("PACF")
                st.pyplot(fig_corr)

        with tab3:
            st.subheader("Forecasted Data Table (to 2035)")
            forecast_export = forecast_df.copy()
            forecast_export['Year'] = forecast_export.index.year
            forecast_export['Quarter'] = forecast_export.index.quarter.map(lambda x: f"Q{x}")
            forecast_export = forecast_export[['Year', 'Quarter', 'Forecast']].rename(columns={'Forecast': 'Predicted_Production_MT'})
            
            st.dataframe(forecast_export.style.format({'Predicted_Production_MT': '{:.2f}'}), use_container_width=True)
            
            csv = forecast_export.to_csv().encode('utf-8')
            st.download_button("Download Forecast CSV", csv, f"forecast_{selected_brgy}.csv", "text/csv")

    else:
        st.info("Please upload the Copra Production CSV file to begin.")
        st.image("https://via.placeholder.com/800x400.png?text=Waiting+for+Data+Upload", use_container_width=True)

if __name__ == "__main__":
    main()
