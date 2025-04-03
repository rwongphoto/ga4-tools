import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import math
from datetime import timedelta

def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.info("Awaiting CSV file upload...")
        return None

def plot_daily_forecast(df, forecast_end_date):
    # Convert dates and rename columns for Prophet
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    last_date = df['ds'].max()
    
    # Compute forecast periods as the number of days from last observed date
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date for daily forecast.")
        return None, last_date

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
    
    ax.set_title('Daily Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    st.pyplot(fig)
    
    return forecast, last_date

def plot_weekly_forecast(df, forecast_end_date):
    # Process and aggregate data by week
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    df.set_index('ds', inplace=True)
    df_weekly = df.resample('W').sum().reset_index()
    last_date = df_weekly['ds'].max()
    
    # Compute forecast periods as the number of weeks from last observed date
    periods = math.ceil((forecast_end_date - last_date).days / 7)
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date for weekly forecast.")
        return None, last_date

    model = Prophet()
    model.fit(df_weekly)
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    
    # Plot the weekly data
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df_weekly['ds'], df_weekly['y'], label='Weekly Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Weekly Forecast', color='green')
    
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
        ('20241219', '20241226', 'Dec 2024 Spam Update')
    ]
    for start_str, end_str, label in google_updates:
        start_date = pd.to_datetime(start_str, format='%Y%m%d')
        end_date = pd.to_datetime(end_str, format='%Y%m%d')
        ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
        mid_date = start_date + (end_date - start_date) / 2
        ax.text(mid_date, ax.get_ylim()[1], label, ha='center', va='top', fontsize=9)
    
    ax.set_title('Weekly Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions (Weekly)')
    ax.legend()
    st.pyplot(fig)
    
    return forecast, last_date

def plot_monthly_forecast(df, forecast_end_date):
    # Process and aggregate data by month
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    df.set_index('ds', inplace=True)
    df_monthly = df.resample('M').sum().reset_index()
    last_date = df_monthly['ds'].max()
    
    # Calculate number of months to forecast
    months_diff = (forecast_end_date.year - last_date.year) * 12 + (forecast_end_date.month - last_date.month)
    if forecast_end_date.day > last_date.day:
        months_diff += 1
    periods = months_diff
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date for monthly forecast.")
        return None, last_date

    model = Prophet()
    model.fit(df_monthly)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    
    # Plot the monthly data
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df_monthly['ds'], df_monthly['y'], label='Monthly Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Monthly Forecast', color='green')
    
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
        ('20241219', '20241226', 'Dec 2024 Spam Update')
    ]
    for start_str, end_str, label in google_updates:
        start_date = pd.to_datetime(start_str, format='%Y%m%d')
        end_date = pd.to_datetime(end_str, format='%Y%m%d')
        ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
        mid_date = start_date + (end_date - start_date) / 2
        ax.text(mid_date, ax.get_ylim()[1], label, ha='center', va='top', fontsize=9)
    
    ax.set_title('Monthly Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions (Monthly)')
    ax.legend()
    st.pyplot(fig)
    
    return forecast, last_date

def display_dashboard(forecast, last_date, forecast_end_date, forecast_type):
    st.subheader("Forecast Data Table")
    # Show forecast rows between the last observed date and the selected forecast end date
    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]
    st.dataframe(forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    # Compute forecast horizon depending on forecast type
    if forecast_type == "Daily Forecast":
        horizon = (forecast_end_date - last_date).days
        horizon_str = f"{horizon} days"
    elif forecast_type == "Weekly Forecast":
        horizon = math.ceil((forecast_end_date - last_date).days / 7)
        horizon_str = f"{horizon} weeks"
    else:  # Monthly Forecast
        months_diff = (forecast_end_date.year - last_date.year) * 12 + (forecast_end_date.month - last_date.month)
        if forecast_end_date.day > last_date.day:
            months_diff += 1
        horizon = months_diff
        horizon_str = f"{horizon} months"
    
    # Find the forecast row closest to the forecast end date
    forecast_future = forecast[forecast['ds'] > last_date]
    if forecast_future.empty:
        st.write("No forecast data available for the selected date range.")
        return
    closest_idx = (forecast_future['ds'] - forecast_end_date).abs().idxmin()
    forecast_value = forecast_future.loc[closest_idx]
    
    st.subheader("Forecast Summary")
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon_str}")
    st.metric(label="Forecasted Traffic", value=int(forecast_value['yhat']),
              delta=f"{int(forecast_value['yhat_upper'] - forecast_value['yhat_lower'])} range")

def main():
    st.set_page_config(page_title="Google Algorithm Update Forecast", layout="wide")
    st.title("Google Algorithm Update Forecast")
    st.write("""
        This app loads GA4 data, fits a Prophet model to forecast future sessions,
        and displays actual vs. forecasted traffic with shaded Google update ranges.
        Choose a forecast type (Daily, Weekly, or Monthly) and select a forecast end date.
        A table and summary dashboard are provided underneath the chart.
    """)
    
    # Sidebar: choose forecast type and set forecast end date
    forecast_type = st.sidebar.radio("Select Forecast Type", ("Daily Forecast", "Weekly Forecast", "Monthly Forecast"))
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end)
    forecast_end_date = pd.to_datetime(forecast_end_date_input)
    
    df = load_data()
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        if forecast_type == "Daily Forecast":
            forecast, last_date = plot_daily_forecast(df.copy(), forecast_end_date)
        elif forecast_type == "Weekly Forecast":
            forecast, last_date = plot_weekly_forecast(df.copy(), forecast_end_date)
        else:
            forecast, last_date = plot_monthly_forecast(df.copy(), forecast_end_date)
        
        if forecast is not None:
            display_dashboard(forecast, last_date, forecast_end_date, forecast_type)

if __name__ == "__main__":
    main()
