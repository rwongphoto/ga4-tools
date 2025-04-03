import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import math # Needed for ceiling function

def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Basic validation for required columns
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("CSV must contain 'Date' and 'Sessions' columns.")
                return None
            # Attempt to parse date early to catch format errors
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            # Ensure Sessions is numeric
            df['Sessions'] = pd.to_numeric(df['Sessions'], errors='coerce')
            df.dropna(subset=['Sessions'], inplace=True) # Drop rows where Sessions couldn't be converted
            return df
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    else:
        st.info("Awaiting CSV file upload...")
        return None

def plot_forecast(df, forecast, google_updates, frequency):
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot actual data points for clarity, especially with aggregation
    ax.plot(df['ds'], df['y'], 'o-', label='Actual', color='blue', markersize=4, linewidth=1)
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green', linewidth=2)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='green', alpha=0.2, label='Forecast Uncertainty')


    # Add Google Update ranges
    y_min, y_max = ax.get_ylim() # Get current y-limits after plotting data
    text_y_position = y_max * 0.98 # Position text near the top

    for start_str, end_str, label in google_updates:
        try:
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            # Only plot updates within the plot's date range for clarity
            if start_date <= forecast['ds'].max() and end_date >= df['ds'].min():
                 ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
                 mid_date = start_date + (end_date - start_date) / 2
                 # Check if mid_date is within the plot limits before adding text
                 if mid_date >= ax.get_xlim()[0] and mid_date <= ax.get_xlim()[1]:
                     ax.text(mid_date, text_y_position, label, ha='center', va='top', fontsize=8, rotation=0,
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none')) # Add background to text
        except ValueError:
            st.warning(f"Could not parse date for update: {label}. Skipping.")


    # Dynamic Titles and Labels
    freq_str_map = {'D': 'Daily', 'W': 'Weekly', 'MS': 'Monthly'}
    plot_title = f'Actual vs. Forecasted {freq_str_map.get(frequency, frequency)} GA4 Sessions with Google Update Ranges'
    y_label = f'{freq_str_map.get(frequency, frequency)} Sessions'

    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout() # Adjust plot to prevent labels overlapping
    st.pyplot(fig)

def main():
    st.set_page_config(layout="wide") # Use wider layout
    st.title("GA4 Forecasting with Prophet")
    st.write("""
        Upload your GA4 data (CSV with 'Date' in YYYYMMDD format and 'Sessions' columns).
        Select the desired forecast frequency (Daily, Weekly, or Monthly).
        The app fits a Prophet model to forecast future sessions and displays results
        along with Google algorithm update periods.
    """)

    df_original = load_data()

    if df_original is not None:
        st.write("Original Data Preview (first 5 rows):")
        st.dataframe(df_original.head())

        # --- User Selection ---
        forecast_frequency = st.selectbox(
            "Select Forecast Frequency:",
            ('Daily', 'Weekly', 'Monthly'),
            index=0 # Default to Daily
        )

        # --- Data Preparation based on Frequency ---
        df_processed = df_original.copy()
        df_processed.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
        df_processed = df_processed[['ds', 'y']].sort_values('ds') # Ensure correct columns and sorting

        prophet_freq = 'D' # Default
        forecast_periods = 90 # Default

        if forecast_frequency == 'Weekly':
            prophet_freq = 'W'
            # Aggregate data to weekly frequency (Week starting Sunday)
            # Ensure ds is the index for resampling
            df_processed.set_index('ds', inplace=True)
            df_processed = df_processed.resample('W').sum().reset_index()
            # Adjust periods: 90 days is roughly 13 weeks
            forecast_periods = math.ceil(90 / 7)
            st.write(f"Data Aggregated to Weekly Frequency (Sum of Sessions):")
            st.dataframe(df_processed.head())

        elif forecast_frequency == 'Monthly':
            prophet_freq = 'MS' # Month Start frequency
            # Aggregate data to monthly frequency
            # Ensure ds is the index for resampling
            df_processed.set_index('ds', inplace=True)
            df_processed = df_processed.resample('MS').sum().reset_index()
            # Adjust periods: 90 days is roughly 3 months
            forecast_periods = math.ceil(90 / 30)
            st.write(f"Data Aggregated to Monthly Frequency (Sum of Sessions):")
            st.dataframe(df_processed.head())
        else: # Daily
             st.write("Using Daily Frequency.")
             st.dataframe(df_processed.head()) # Show the renamed daily data


        # --- Check if enough data points for the chosen frequency ---
        min_data_points = 5 # Arbitrary minimum for Prophet to work reasonably
        if len(df_processed) < min_data_points:
             st.warning(f"Warning: Only {len(df_processed)} data points after processing for {forecast_frequency} frequency. Forecast might be unreliable.")
             st.stop() # Stop execution if data is too sparse


        # --- Fit the Prophet model ---
        st.write(f"Fitting Prophet model for {forecast_frequency} forecast...")
        model = Prophet()
        try:
            model.fit(df_processed)
        except Exception as e:
            st.error(f"Error fitting Prophet model: {e}")
            st.error("This might happen with very sparse data after aggregation. Try using 'Daily' frequency or a dataset with more history.")
            st.stop()


        # --- Create Future Dataframe & Forecast ---
        st.write(f"Generating forecast for the next {forecast_periods} periods ({forecast_frequency})...")
        future = model.make_future_dataframe(periods=forecast_periods, freq=prophet_freq)
        try:
            forecast = model.predict(future)
        except Exception as e:
             st.error(f"Error generating forecast: {e}")
             st.stop()

        st.write("Forecast Data Preview (last 5 periods):")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # --- Define Google algorithm update ranges ---
        # (Keep these as daily ranges, axvspan works with dates)
        google_updates = [
            ('20230315', '20230328', 'Mar 23 Core'),
            ('20230822', '20230907', 'Aug 23 Core'),
            ('20230914', '20230928', 'Sept 23 HCU'),
            ('20231004', '20231019', 'Oct 23 Core/Spam'),
            ('20231102', '20231204', 'Nov 23 Core/Spam'),
            ('20240305', '20240419', 'Mar 24 Core'),
            ('20240506', '20240507', 'Site Rep Abuse'),
            # ('20240514', '20240515', 'AI Overviews'), # Often too short to see clearly
            ('20240620', '20240627', 'Jun 24 Core'),
            ('20240815', '20240903', 'Aug 24 Core'),
            ('20241111', '20241205', 'Nov 24 Core'), # Future dates
            ('20241212', '20241218', 'Dec 24 Core'), # Future dates
            ('20241219', '20241226', 'Dec 24 Spam'), # Future dates
            ('20250313', '20250327', 'Mar 25 Core')  # Future dates
        ]

        # --- Plot the forecast results ---
        st.subheader(f"{forecast_frequency} Forecast Plot")
        plot_forecast(df_processed, forecast, google_updates, prophet_freq)

        # --- Display Components (Optional) ---
        st.subheader(f"{forecast_frequency} Forecast Components")
        try:
            fig_comp = model.plot_components(forecast)
            st.pyplot(fig_comp)
        except Exception as e:
            st.warning(f"Could not plot components: {e}")


if __name__ == "__main__":
    main()
