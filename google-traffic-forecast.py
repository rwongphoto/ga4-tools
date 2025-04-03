import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from neuralprophet import NeuralProphet
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from datetime import timedelta

def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.info("Awaiting CSV file upload...")
        return None

def plot_daily_forecast(df, forecast_end_date, model_selection):
    # Convert 'Date' column from string format (YYYYMMDD) to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    
    last_date = df['Date'].max()
    
    # Calculate forecast periods as the number of days from the last observed date
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date for daily forecast.")
        return None, last_date

    # Create a copy of dataframe for forecasting
    forecast_df = df.copy()
    
    # Apply the selected forecasting model
    if model_selection == 'Prophet':
        forecast, future = prophet_forecast(forecast_df, periods)
    elif model_selection == 'NeuralProphet':
        forecast, future = neuralprophet_forecast(forecast_df, periods)
    elif model_selection == 'StatsForecast AutoARIMA':
        forecast, future = statsforecast_arima(forecast_df, periods)
    elif model_selection == 'StatsForecast AutoETS':
        forecast, future = statsforecast_ets(forecast_df, periods)
    else:
        st.error("Invalid model selection")
        return None, last_date
    
    # Plot actual vs. forecast
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['Date'], df['Sessions'], label='Actual', color='blue')
    
    # For models with different output formats
    if model_selection == 'Prophet':
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')
    elif model_selection == 'NeuralProphet':
        ax.plot(forecast['ds'], forecast['yhat1'], label='Forecast', color='green')
    elif model_selection.startswith('StatsForecast'):
        # Combine historical and forecast data
        complete_forecast = pd.concat([
            pd.DataFrame({'ds': df['Date'], 'yhat': df['Sessions']}),
            pd.DataFrame({'ds': future.index, 'yhat': future['mean']})
        ])
        ax.plot(complete_forecast['ds'], complete_forecast['yhat'], label='Forecast', color='green')
    
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
    
    ax.set_title(f'Daily Actual vs. Forecasted GA4 Sessions with {model_selection} Model')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    st.pyplot(fig)
    
    return forecast, last_date

def prophet_forecast(df, periods):
    # Rename columns for Prophet
    prophet_df = df.rename(columns={'Date': 'ds', 'Sessions': 'y'})
    
    # Fit the Prophet model
    with st.spinner('Training Prophet model...'):
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, freq='D')
        forecast = model.predict(future)
    
    return forecast, None

def neuralprophet_forecast(df, periods):
    # Rename columns for NeuralProphet
    np_df = df.rename(columns={'Date': 'ds', 'Sessions': 'y'})
    
    # Fit the NeuralProphet model
    with st.spinner('Training NeuralProphet model (this may take a while)...'):
        model = NeuralProphet()
        model.fit(np_df, freq="D")
        future = model.make_future_dataframe(np_df, periods=periods)
        forecast = model.predict(future)
    
    return forecast, None

def statsforecast_arima(df, periods):
    # Prepare data for StatsForecast
    sf_df = df.rename(columns={'Date': 'ds', 'Sessions': 'y'})
    sf_df['unique_id'] = 'sessions'  # StatsForecast requires a unique_id column
    sf_df = sf_df.set_index('ds')[['unique_id', 'y']]
    
    # Define model
    with st.spinner('Training StatsForecast AutoARIMA model...'):
        models = [AutoARIMA(season_length=7)]  # Weekly seasonality
        sf = StatsForecast(
            models=models,
            freq='D',
            n_jobs=-1  # Use all available cores
        )
        
        # Fit and predict
        sf.fit(sf_df)
        forecast = sf.predict(h=periods)
    
    return None, forecast

def statsforecast_ets(df, periods):
    # Prepare data for StatsForecast
    sf_df = df.rename(columns={'Date': 'ds', 'Sessions': 'y'})
    sf_df['unique_id'] = 'sessions'  # StatsForecast requires a unique_id column
    sf_df = sf_df.set_index('ds')[['unique_id', 'y']]
    
    # Define model
    with st.spinner('Training StatsForecast ETS model...'):
        models = [AutoETS(season_length=7)]  # Weekly seasonality
        sf = StatsForecast(
            models=models,
            freq='D',
            n_jobs=-1  # Use all available cores
        )
        
        # Fit and predict
        sf.fit(sf_df)
        forecast = sf.predict(h=periods)
    
    return None, forecast

def display_dashboard(forecast, last_date, forecast_end_date, model_selection):
    st.subheader("Forecast Data Table")
    
    if model_selection == 'Prophet':
        # Display forecast rows between the last observed date and the forecast end date
        forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]
        st.dataframe(forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        # Year-over-Year Calculation
        start_forecast = last_date + pd.Timedelta(days=1)
        end_forecast = forecast_end_date
        current_period = forecast[(forecast['ds'] >= start_forecast) & (forecast['ds'] <= end_forecast)]
        # Define the corresponding period one year earlier
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
            
        # Get the forecast row closest to the forecast end date
        forecast_future = forecast[forecast['ds'] > last_date]
        closest_idx = (forecast_future['ds'] - forecast_end_date).abs().idxmin()
        forecast_value = forecast_future.loc[closest_idx]
        st.metric(label="Forecasted Traffic", value=int(forecast_value['yhat']),
                delta=f"Range: {int(forecast_value['yhat_upper'] - forecast_value['yhat_lower'])}")
    
    elif model_selection == 'NeuralProphet':
        # Display forecast rows between the last observed date and the forecast end date
        forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]
        st.dataframe(forecast_filtered[['ds', 'yhat1']])
        
        # Get the forecast row closest to the forecast end date
        forecast_future = forecast[forecast['ds'] > last_date]
        closest_idx = (forecast_future['ds'] - forecast_end_date).abs().idxmin()
        forecast_value = forecast_future.loc[closest_idx]
        st.metric(label="Forecasted Traffic", value=int(forecast_value['yhat1']), delta=None)
    
    elif model_selection.startswith('StatsForecast'):
        # For StatsForecast models
        # The forecast object here is the future dataframe with predictions
        st.dataframe(forecast)
        
        # Calculate summary metrics
        forecast_sum = forecast['mean'].sum()
        st.metric(label="Total Forecasted Traffic", value=int(forecast_sum), 
                 delta=f"Range: {int(forecast['mean'].max() - forecast['mean'].min())}")
    
    # Calculate forecast horizon
    horizon = (forecast_end_date - last_date).days
    st.subheader("Forecast Summary")
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon} days")
    st.write(f"Forecasting Model: {model_selection}")

def main():
    st.title("GA4 Daily Forecasting with Multiple Models")
    st.write("""
        This app loads GA4 data, fits your selected forecasting model to predict daily sessions,
        and displays actual vs. forecasted traffic with shaded Google update ranges.
        A summary dashboard with forecast metrics is provided below.
        The CSV file must have a column for "Date" and one for "Sessions". Date should be sorted by oldest date first.
    """)
    
    # Sidebar: model selection and forecast end date
    model_selection = st.sidebar.selectbox(
        "Select Forecasting Model",
        ["Prophet", "NeuralProphet", "StatsForecast AutoARIMA", "StatsForecast AutoETS"],
        index=0
    )
    
    # Model information tooltips
    if model_selection == "Prophet":
        st.sidebar.info("Facebook's Prophet is good for data with strong seasonal patterns and multiple seasonalities.")
    elif model_selection == "NeuralProphet":
        st.sidebar.info("NeuralProphet combines Neural Networks with Prophet. It generally handles complex patterns better but may take longer to train.")
    elif model_selection == "StatsForecast AutoARIMA":
        st.sidebar.info("AutoARIMA automatically selects the best ARIMA parameters. Good for data with trends and seasonality.")
    elif model_selection == "StatsForecast AutoETS":
        st.sidebar.info("AutoETS automatically selects the best Exponential Smoothing parameters. Good for data with trend and seasonality but no complex patterns.")
    
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end)
    forecast_end_date = pd.to_datetime(forecast_end_date_input)
    
    # Advanced settings (optional)
    with st.sidebar.expander("Advanced Settings"):
        if model_selection == "NeuralProphet":
            st.number_input("Epochs", min_value=10, max_value=500, value=100, step=10, 
                            help="More epochs may improve accuracy but increase training time")
        elif model_selection.startswith("StatsForecast"):
            st.number_input("Season Length", min_value=1, max_value=30, value=7, 
                            help="Typically 7 for weekly patterns or 30 for monthly patterns")
    
    # Load GA4 data
    df = load_data()
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        forecast, last_date = plot_daily_forecast(df.copy(), forecast_end_date, model_selection)
        if forecast is not None:
            display_dashboard(forecast, last_date, forecast_end_date, model_selection)
            
            # Option to download the forecast numbers as CSV
            if model_selection == "Prophet":
                csv_data = forecast.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Forecast CSV",
                    data=csv_data,
                    file_name=f'{model_selection}_forecast.csv',
                    mime='text/csv'
                )
            elif model_selection == "NeuralProphet":
                csv_data = forecast.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Forecast CSV",
                    data=csv_data,
                    file_name=f'{model_selection}_forecast.csv',
                    mime='text/csv'
                )
            elif model_selection.startswith("StatsForecast"):
                csv_data = forecast.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Full Forecast CSV",
                    data=csv_data,
                    file_name=f'{model_selection}_forecast.csv',
                    mime='text/csv'
                )
    
    # Footer link
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()

