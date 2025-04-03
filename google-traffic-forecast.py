import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Import Prophet (used by Facebook Prophet)
from prophet import Prophet
# Note: Ensure NeuralProphet and statsforecast libraries are installed.

def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.info("Awaiting CSV file upload...")
        return None

def forecast_daily_prophet(df, forecast_end_date):
    # Process data for Prophet
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    last_date = df['ds'].max()
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date.")
        return None, last_date, df
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    
    # Plot actual vs. forecast
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['ds'], df['y'], label='Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')
    
    # Shade Google algorithm update ranges
    google_updates = [
        ('20230315', '20230328', 'Mar 2023 Core Update'),
        ('20230822', '20230907', 'Aug 2023 Core Update'),
        ('20230914', '20230928', 'Sept 2023 Helpful Content Update'),
        ('20231004', '20231019', 'Oct 2023 Core & Spam Updates'),
        ('20231102', '20231204', 'Nov 2023 Core & Spam Updates'),
        ('20240305', '20240419', 'Mar 2024 Core Update'),
        ('20240506', '20240507', 'Site Rep Abuse'),
        ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'June 2024 Core Update'),
        ('20240815', '20240903', 'Aug 2024 Core Update'),
        ('20241111', '20241205', 'Nov 2024 Core Update'),
        ('20241212', '20241218', 'Dec 2024 Core Update'),
        ('20241219', '20241226', 'Dec 2024 Spam Update'),
        ('20250313', '20250327', 'Mar 2025 Core Update')
    ]
    for start_str, end_str, label in google_updates:
        start_date = pd.to_datetime(start_str, format='%Y%m%d')
        end_date = pd.to_datetime(end_str, format='%Y%m%d')
        ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
        mid_date = start_date + (end_date - start_date) / 2
        ax.text(mid_date, ax.get_ylim()[1], label, ha='center', va='top', fontsize=9)
    ax.set_title('Daily Forecast (Prophet)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    st.pyplot(fig)
    
    return forecast, last_date, df

def forecast_daily_neuralprophet(df, forecast_end_date):
    from neuralprophet import NeuralProphet
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    last_date = df['ds'].max()
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date.")
        return None, last_date, df
    model = NeuralProphet()
    model.fit(df, freq='D', progress="silent")
    future = model.make_future_dataframe(df, periods=periods)
    forecast = model.predict(future)
    # NeuralProphet outputs predictions in 'yhat1'
    if 'yhat1' in forecast.columns:
        forecast = forecast.rename(columns={'yhat1': 'yhat'})
    # Plot actual vs. forecast
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['ds'], df['y'], label='Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')
    for start_str, end_str, label in [
        ('20230315', '20230328', 'Mar 2023 Core Update'),
        ('20230822', '20230907', 'Aug 2023 Core Update'),
        ('20230914', '20230928', 'Sept 2023 Helpful Content Update'),
        ('20231004', '20231019', 'Oct 2023 Core & Spam Updates'),
        ('20231102', '20231204', 'Nov 2023 Core & Spam Updates'),
        ('20240305', '20240419', 'Mar 2024 Core Update'),
        ('20240506', '20240507', 'Site Rep Abuse'),
        ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'June 2024 Core Update'),
        ('20240815', '20240903', 'Aug 2024 Core Update'),
        ('20241111', '20241205', 'Nov 2024 Core Update'),
        ('20241212', '20241218', 'Dec 2024 Core Update'),
        ('20241219', '20241226', 'Dec 2024 Spam Update'),
        ('20250313', '20250327', 'Mar 2025 Core Update')
    ]:
        start_date = pd.to_datetime(start_str, format='%Y%m%d')
        end_date = pd.to_datetime(end_str, format='%Y%m%d')
        ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
        mid_date = start_date + (end_date - start_date) / 2
        ax.text(mid_date, ax.get_ylim()[1], label, ha='center', va='top', fontsize=9)
    ax.set_title('Daily Forecast (NeuralProphet)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    st.pyplot(fig)
    
    return forecast, last_date, df

def forecast_daily_statsforecast(df, forecast_end_date):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    df['unique_id'] = 'GA4'
    df = df[['unique_id', 'ds', 'y']]
    last_date = df['ds'].max()
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date.")
        return None, last_date, df
    sf = StatsForecast(df=df, models=[AutoARIMA()], freq='D')
    forecast_df = sf.forecast(horizon=periods)
    # Rename the forecast column (named after the model) to 'yhat'
    model_col = forecast_df.columns[-1]
    forecast_df = forecast_df.rename(columns={model_col: 'yhat'})
    # Plot actual vs. forecast
    fig, ax = plt.subplots(figsize=(16, 8))
    actual_df = df[df['unique_id'] == 'GA4']
    ax.plot(actual_df['ds'], actual_df['y'], label='Actual', color='blue')
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='green')
    for start_str, end_str, label in [
        ('20230315', '20230328', 'Mar 2023 Core Update'),
        ('20230822', '20230907', 'Aug 2023 Core Update'),
        ('20230914', '20230928', 'Sept 2023 Helpful Content Update'),
        ('20231004', '20231019', 'Oct 2023 Core & Spam Updates'),
        ('20231102', '20231204', 'Nov 2023 Core & Spam Updates'),
        ('20240305', '20240419', 'Mar 2024 Core Update'),
        ('20240506', '20240507', 'Site Rep Abuse'),
        ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'June 2024 Core Update'),
        ('20240815', '20240903', 'Aug 2024 Core Update'),
        ('20241111', '20241205', 'Nov 2024 Core Update'),
        ('20241212', '20241218', 'Dec 2024 Core Update'),
        ('20241219', '20241226', 'Dec 2024 Spam Update'),
        ('20250313', '20250327', 'Mar 2025 Core Update')
    ]:
        start_date = pd.to_datetime(start_str, format='%Y%m%d')
        end_date = pd.to_datetime(end_str, format='%Y%m%d')
        ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
        mid_date = start_date + (end_date - start_date) / 2
        ax.text(mid_date, ax.get_ylim()[1], label, ha='center', va='top', fontsize=9)
    ax.set_title('Daily Forecast (Nixtla Statsforecast)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    st.pyplot(fig)
    
    return forecast_df, last_date, df

def display_dashboard(forecast, last_date, forecast_end_date):
    st.subheader("Forecast Data Table")
    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]
    if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
        st.dataframe(forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    else:
        st.dataframe(forecast_filtered[['ds', 'yhat']])
    
    horizon = (forecast_end_date - last_date).days
    st.subheader("Forecast Summary")
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon} days")
    forecast_future = forecast[forecast['ds'] > last_date]
    if forecast_future.empty:
        st.write("No forecast data available for the selected date range.")
        return
    closest_idx = (forecast_future['ds'] - forecast_end_date).abs().idxmin()
    forecast_value = forecast_future.loc[closest_idx]
    st.metric(label="Forecasted Traffic", value=int(forecast_value['yhat']),
              delta=f"Range: {int(forecast_value.get('yhat_upper', 0) - forecast_value.get('yhat_lower', 0))}")
    
    # Year-over-Year Calculation
    start_forecast = last_date + pd.Timedelta(days=1)
    end_forecast = forecast_end_date
    current_period = forecast[(forecast['ds'] >= start_forecast) & (forecast['ds'] <= end_forecast)]
    start_prev = start_forecast - pd.Timedelta(days=365)
    end_prev = end_forecast - pd.Timedelta(days=365)
    prev_period = forecast[(forecast['ds'] >= start_prev) & (forecast['ds'] <= end_prev)]
    if not current_period.empty and not prev_period.empty:
        current_sum = current_period['yhat'].sum()
        prev_sum = prev_period['yhat'].sum()
        if prev_sum != 0:
            yoy_change = ((current_sum - prev_sum) / prev_sum) * 100
        else:
            yoy_change = float('inf')
        st.subheader("Year-over-Year Comparison")
        st.write(f"Total Forecasted Traffic for Selected Period: {current_sum:.0f}")
        st.write(f"Total Traffic for Same Period Last Year: {prev_sum:.0f}")
        st.write(f"Year-over-Year Change: {yoy_change:.2f}%")
    else:
        st.write("Not enough data for Year-over-Year calculation.")

def main():
    st.title("GA4 Daily Forecasting with Multiple Models")
    st.write("""
        This app loads GA4 data, fits a forecasting model to predict daily sessions,
        and displays actual vs. forecasted traffic with shaded Google update ranges.
        Choose a forecasting model, select a forecast end date, and view the summary dashboard.
    """)
    
    # Sidebar: select forecasting model and forecast end date
    model_option = st.sidebar.selectbox("Select Forecasting Model", ["Prophet", "NeuralProphet", "Nixtla Statsforecast"])
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end)
    forecast_end_date = pd.to_datetime(forecast_end_date_input)
    
    df = load_data()
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        forecast = None
        last_date = None
        
        if model_option == "Prophet":
            forecast, last_date, _ = forecast_daily_prophet(df.copy(), forecast_end_date)
        elif model_option == "NeuralProphet":
            forecast, last_date, _ = forecast_daily_neuralprophet(df.copy(), forecast_end_date)
        else:  # Nixtla Statsforecast
            forecast, last_date, _ = forecast_daily_statsforecast(df.copy(), forecast_end_date)
        
        if forecast is not None:
            display_dashboard(forecast, last_date, forecast_end_date)
            csv_data = forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Forecast CSV",
                data=csv_data,
                file_name='forecast.csv',
                mime='text/csv'
            )
    
    st.markdown("[Created by The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()

