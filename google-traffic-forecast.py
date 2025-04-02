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

def plot_forecast(df, forecast, google_updates):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['ds'], df['y'], label='Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')

    for start_str, end_str, label in google_updates:
        start_date = pd.to_datetime(start_str, format='%Y%m%d')
        end_date = pd.to_datetime(end_str, format='%Y%m%d')
        ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
        mid_date = start_date + (end_date - start_date) / 2
        # Position the label at the top of the plot.
        ax.text(mid_date, ax.get_ylim()[1], label, ha='center', va='top', fontsize=9)
    
    ax.set_title('Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("GA4 Forecasting with Prophet")
    st.write("""
        This app loads GA4 data, fits a Prophet model to forecast future sessions, 
        and displays both the actual and forecasted values along with shaded regions 
        representing Google algorithm update periods. You will need one column for Date and one column for Sessions.
    """)

    # Load data interactively.
    df = load_data()
    if df is not None:
        st.write("Original Data Preview:")
        st.write(df.head())
        
        # Convert and rename columns as required by Prophet.
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
        st.write("Data After Converting Date:")
        st.write(df.head())
        
        # Fit the Prophet model.
        model = Prophet()
        model.fit(df)
        
        # Forecast the next 90 days.
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)
        
        # Define Google algorithm update ranges.
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
        
        # Plot the forecast results.
        plot_forecast(df, forecast, google_updates)

if __name__ == "__main__":
    main()
