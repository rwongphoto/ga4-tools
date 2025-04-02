import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.info("Awaiting CSV file upload...")
        return None

def plot_daily_forecast(df):
    # Convert the 'Date' column from string format (YYYYMMDD) to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    # Rename columns to match Prophet's expected names.
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    
    # Fit the Prophet model on daily data
    model = Prophet()
    model.fit(df)
    
    # Forecast for an additional 90 days.
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['ds'], df['y'], label='Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')
    
    # Define Google update ranges for the daily chart.
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
    
    ax.set_title('Actual vs. Forecasted GA4 Sessions with Google Update Ranges (Daily)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    st.pyplot(fig)

def plot_weekly_forecast(df):
    # Convert the 'Date' column from string format (YYYYMMDD) to datetime.
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    # Rename columns for Prophet.
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    
    # Aggregate data by week (summing sessions).
    df.set_index('ds', inplace=True)
    df_weekly = df.resample('W').sum().reset_index()
    
    # Fit the Prophet model on weekly data.
    model = Prophet()
    model.fit(df_weekly)
    
    # Forecast for an additional 10 weeks using weekly frequency.
    future = model.make_future_dataframe(periods=10, freq='W')
    forecast = model.predict(future)
    
    # Create the plot.
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df_weekly['ds'], df_weekly['y'], label='Weekly Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Weekly Forecast', color='green')
    
    # Define Google update ranges for the weekly chart.
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
    
    ax.set_title('Weekly Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions (Weekly)')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("GA4 Forecasting with Prophet")
    st.write("""
        This app loads GA4 data, fits a Prophet model to forecast future sessions, 
        and displays both the actual and forecasted values along with shaded regions 
        representing Google algorithm update periods dating back to the January 2023.
    """)
    
    # Sidebar selection for forecast type.
    forecast_type = st.sidebar.radio("Select Forecast Type", ("Daily Forecast", "Weekly Forecast"))
    
    # Load GA4 data.
    df = load_data()
    
    if df is not None:
        st.write("Data Preview:")
        st.write(df.head())
        
        if forecast_type == "Daily Forecast":
            plot_daily_forecast(df.copy())
        else:
            plot_weekly_forecast(df.copy())

if __name__ == "__main__":
    main()

