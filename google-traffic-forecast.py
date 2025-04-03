import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import torch
from neuralprophet import NeuralProphet
from neuralprophet.configure import ConfigSeasonality

# Allow the global used by NeuralProphet to be loaded safely.
torch.serialization.add_safe_global("neuralprophet.configure.ConfigSeasonality", ConfigSeasonality)

def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.info("Awaiting CSV file upload...")
        return None

def forecast_daily_neuralprophet(df, forecast_end_date):
    # Convert 'Date' column from string format (YYYYMMDD) to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    # Rename columns for NeuralProphet (expects 'ds' and 'y')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    last_date = df['ds'].max()
    
    # Compute the number of days to forecast
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date.")
        return None, last_date, df
    
    # Create and fit the NeuralProphet model
    model = NeuralProphet()
    model.fit(df, freq='D', progress='silent')
    
    # Create a future dataframe and predict
    future = model.make_future_dataframe(df, periods=periods)
    forecast = model.predict(future)
    
    # NeuralProphet outputs predictions in the 'yhat1' column; rename for consistency
    if 'yhat1' in forecast.columns:
        forecast = forecast.rename(columns={'yhat1': 'yhat'})
    
    # Plot actual vs. forecasted sessions
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
    
    ax.set_title('Daily Forecast (NeuralProphet)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    st.pyplot(fig)
    
    return forecast, last_date, df

def display_dashboard(forecast, last_date, forecast_end_date):
    st.subheader("Forecast Data Table")
    # Show forecast rows between the last observed date and the selected forecast end date
    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]
    st.dataframe(forecast_filtered[['ds', 'yhat']])
    
    # Calculate forecast horizon
    horizon = (forecast_end_date - last_date).days
    st.subheader("Forecast Summary")
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon} days")
    
    # Get the forecast row closest to the forecast end date
    forecast_future = forecast[forecast['ds'] > last_date]
    if forecast_future.empty:
        st.write("No forecast data available for the selected date range.")
        return
    closest_idx = (forecast_future['ds'] - forecast_end_date).abs().idxmin()
    forecast_value = forecast_future.loc[closest_idx]
    st.metric(label="Forecasted Traffic", value=int(forecast_value['yhat']))
    
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
    st.title("GA4 Daily Forecasting with NeuralProphet")
    st.write("""
        This app loads GA4 data, fits a NeuralProphet model to forecast daily sessions,
        and displays actual vs. forecasted traffic with shaded Google update ranges.
        A summary dashboard with a year-over-year comparison is provided below.
    """)
    
    # Sidebar: select forecast end date
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end)
    forecast_end_date = pd.to_datetime(forecast_end_date_input)
    
    # Load GA4 data from CSV file
    df = load_data()
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        forecast, last_date, _ = forecast_daily_neuralprophet(df.copy(), forecast_end_date)
        if forecast is not None:
            display_dashboard(forecast, last_date, forecast_end_date)
            
            # Option to download the full forecast as CSV
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
