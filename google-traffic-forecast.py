import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# Change import from Prophet to NeuralProphet
from neuralprophet import NeuralProphet, set_log_level
from datetime import timedelta
import logging

# Optional: Suppress excessive logging from NeuralProphet during training
set_log_level("ERROR")
# You can also try "WARNING" or "INFO" if you want some feedback
# logging.getLogger("NP").setLevel(logging.ERROR) # Alternative way

def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Basic validation for required columns
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("CSV must contain 'Date' and 'Sessions' columns.")
                return None
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    else:
        st.info("Awaiting CSV file upload...")
        return None

def plot_daily_forecast(df, forecast_end_date):
    try:
        # Convert 'Date' column from string format (YYYYMMDD) to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    except ValueError:
        st.error("Error parsing 'Date' column. Ensure it's in YYYYMMDD format.")
        return None, None
    except Exception as e:
         st.error(f"An error occurred during date conversion: {e}")
         return None, None

    # Sort by date just in case
    df = df.sort_values('Date')

    # Rename columns for NeuralProphet ('ds' for date, 'y' for target)
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)

    # Drop rows where 'y' might be NaN if any exist after rename/load
    df.dropna(subset=['ds', 'y'], inplace=True)
    if df.empty:
        st.error("No valid data remaining after processing. Check your CSV content.")
        return None, None

    last_date = df['ds'].max()

    # Calculate forecast periods as the number of days from the last observed date
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date for daily forecast.")
        return None, last_date

    # --- NeuralProphet Section ---
    # Initialize the NeuralProphet model
    # Common seasonality settings for daily website traffic data
    # quantiles added to get uncertainty intervals similar to Prophet's yhat_lower/upper
    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False, # Usually not needed for daily data
        quantiles=[0.05, 0.95] # For uncertainty intervals
    )

    # Fit the model
    # NeuralProphet requires frequency 'D' for daily data
    # Use a progress bar for potentially longer training
    st.info("Training NeuralProphet model... this may take a few moments.")
    progress_bar = st.progress(0)
    try:
        # Split data for metrics, fit model
        df_train, df_test = model.split_df(df, freq="D", valid_p=0.1) # Use last 10% for validation
        metrics = model.fit(df_train, freq="D")
        progress_bar.progress(100) # Mark as complete after fitting
        st.write("Model Training Metrics (on validation set):")
        st.write(metrics) # Display training metrics
    except Exception as e:
        st.error(f"Error fitting NeuralProphet model: {e}")
        progress_bar.progress(100) # Ensure progress bar finishes on error
        return None, last_date

    # Create future dataframe
    future = model.make_future_dataframe(df=df, periods=periods) # Pass original df to ensure all dates are included

    # Make predictions
    try:
        forecast = model.predict(future)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, last_date
    # --- End NeuralProphet Section ---


    # Check if forecast is empty or missing essential columns
    if forecast is None or forecast.empty or 'ds' not in forecast.columns or 'yhat1' not in forecast.columns:
         st.error("Prediction failed or returned unexpected results.")
         return None, last_date

    # --- Plotting Section (Adjusted for NeuralProphet output) ---
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['ds'], df['y'], label='Actual', color='blue', marker='.', linestyle='-') # Added markers for clarity

    # Plot forecast line using 'yhat1'
    ax.plot(forecast['ds'], forecast['yhat1'], label='Forecast (yhat1)', color='green')

    # Plot uncertainty intervals using quantile columns (adjust names if needed)
    # Expected names: 'yhat1 5.0%' and 'yhat1 95.0%' if quantiles=[0.05, 0.95]
    lower_q_col = f'yhat1 {model.quantiles[0]*100:.1f}%'
    upper_q_col = f'yhat1 {model.quantiles[1]*100:.1f}%'

    if lower_q_col in forecast.columns and upper_q_col in forecast.columns:
        ax.fill_between(forecast['ds'],
                        forecast[lower_q_col],
                        forecast[upper_q_col],
                        color='green', alpha=0.2, label='Uncertainty Interval (90%)')
    else:
        st.warning(f"Could not find uncertainty columns: '{lower_q_col}', '{upper_q_col}'. Plotting without intervals.")


    # --- Google Update Shading (No changes needed here) ---
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
    plot_bottom, plot_top = ax.get_ylim() # Get current y-limits for text placement
    for start_str, end_str, label in google_updates:
        try:
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
            # Place text slightly above the plot top
            mid_date = start_date + (end_date - start_date) / 2
            ax.text(mid_date, plot_top, label, ha='center', va='bottom', fontsize=9, rotation=90) # Rotate text for better visibility
        except Exception as e:
            st.warning(f"Could not plot Google Update '{label}': {e}")

    # Reset y-limits slightly expanded to ensure text fits
    ax.set_ylim(plot_bottom, plot_top * 1.1)

    ax.set_title('Daily Actual vs. Forecasted GA4 Sessions (NeuralProphet) with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    st.pyplot(fig)

    return forecast, last_date

def display_dashboard(forecast, last_date, forecast_end_date):
    st.subheader("Forecast Data Table")

    # Define quantile column names based on model's quantiles
    # Ensure model object is accessible or pass quantiles if needed
    # Assuming default [0.05, 0.95] for simplicity here if model object not available
    lower_q_col = 'yhat1 5.0%' # Adjust if using different quantiles
    upper_q_col = 'yhat1 95.0%' # Adjust if using different quantiles

    # Check if quantile columns exist, otherwise create placeholders or warn
    if lower_q_col not in forecast.columns:
        forecast[lower_q_col] = None # Or np.nan
        st.warning(f"Column '{lower_q_col}' not found in forecast data.")
    if upper_q_col not in forecast.columns:
        forecast[upper_q_col] = None # Or np.nan
        st.warning(f"Column '{upper_q_col}' not found in forecast data.")


    # Display forecast rows between the last observed date and the forecast end date
    # Use 'yhat1' for forecast, and quantile columns for range
    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]

    # Select and rename columns for display
    display_cols = ['ds', 'yhat1', lower_q_col, upper_q_col]
    forecast_display = forecast_filtered[display_cols].rename(columns={
        'yhat1': 'Forecast',
        lower_q_col: 'Lower Bound (5%)',
        upper_q_col: 'Upper Bound (95%)'
    })
    st.dataframe(forecast_display)

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

    # Calculate delta based on quantiles if available
    delta_val = None
    if forecast_value[lower_q_col] is not None and forecast_value[upper_q_col] is not None:
         delta_val = int(forecast_value[upper_q_col] - forecast_value[lower_q_col])
         delta_str = f"Range: {delta_val}"
    else:
        delta_str = "Range N/A"

    st.metric(label="Forecasted Traffic (at End Date)", value=int(forecast_value['yhat1']),
              delta=delta_str)

    # Year-over-Year Calculation (using 'yhat1')
    start_forecast = last_date + pd.Timedelta(days=1)
    end_forecast = forecast_end_date
    current_period = forecast[(forecast['ds'] >= start_forecast) & (forecast['ds'] <= end_forecast)]

    # Define the corresponding period one year earlier - Requires historical forecast data
    start_prev = start_forecast - pd.Timedelta(days=365)
    end_prev = end_forecast - pd.Timedelta(days=365)
    # We need forecast data for the previous year period too.
    # NeuralProphet forecast dataframe usually only contains the future dates requested.
    # For a fair YoY comparison using forecast values, we'd ideally need a forecast
    # that was generated *last year* covering this year's period, or use actuals if available.
    # Here, we attempt to find the corresponding *forecasted* values from the *current* forecast run
    # for the dates exactly one year prior. This might be inaccurate if trends/seasonality changed significantly.
    prev_period = forecast[(forecast['ds'] >= start_prev) & (forecast['ds'] <= end_prev)]

    # Alternative: Use actual historical data for the previous period if available
    # This requires passing the original df (or just its 'y' values) into this function.
    # Example (if original df available as `original_df_with_y`):
    # prev_period_actual = original_df_with_y[(original_df_with_y['ds'] >= start_prev) & (original_df_with_y['ds'] <= end_prev)]
    # if not prev_period_actual.empty:
    #     prev_sum = prev_period_actual['y'].sum()


    if not current_period.empty and not prev_period.empty:
        current_sum = current_period['yhat1'].sum()
        prev_sum = prev_period['yhat1'].sum() # Using forecast for prev year period
        if prev_sum != 0:
            yoy_change = ((current_sum - prev_sum) / prev_sum) * 100
            change_label = f"{yoy_change:.2f}%"
        else:
            yoy_change = float('inf') if current_sum > 0 else 0
            change_label = "inf%" if current_sum > 0 else "0.00%"

        st.subheader("Year-over-Year Comparison (Forecast vs Forecast)")
        st.write(f"Total Forecasted Traffic ({start_forecast.date()} to {end_forecast.date()}): {current_sum:.0f}")
        st.write(f"Total Forecasted Traffic ({start_prev.date()} to {end_prev.date()}): {prev_sum:.0f}")
        st.write(f"Year-over-Year Change: {change_label}")
    else:
        st.warning("Not enough historical forecast data within the current run for Year-over-Year calculation.")
        st.write(f"Required historical forecast range: {start_prev.date()} to {end_prev.date()}")

def main():
    st.set_page_config(layout="wide") # Use wider layout
    st.title("GA4 Daily Forecasting with NeuralProphet")
    st.write("""
        This app loads GA4 data (CSV), fits a **NeuralProphet** model to forecast daily sessions,
        and displays actual vs. forecasted traffic with shaded Google update ranges.
        A summary dashboard with a year-over-year comparison is provided below.

        **Requirements:**
        - CSV file must have columns named "Date" (format YYYYMMDD) and "Sessions".
        - Data should be sorted chronologically (oldest first is best).
        - Ensure you have the `neuralprophet` library installed (`pip install neuralprophet`).
    """)

    # Sidebar: set forecast end date
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end, min_value=pd.Timestamp.today().date())
    forecast_end_date = pd.to_datetime(forecast_end_date_input)

    # Load GA4 data
    df = load_data()
    if df is not None:
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df.head())

        # Add a check for minimum data length
        if len(df) < 14: # Need at least some data for weekly seasonality
             st.warning(f"Warning: Very short dataset ({len(df)} rows). NeuralProphet performance might be poor.")
             # Optionally disable seasonality if data is too short
             # model = NeuralProphet(yearly_seasonality=False, weekly_seasonality=False, ...)

        # Plot forecast and display dashboard
        # Pass a copy of df to avoid modifying the original previewed df
        forecast_df, last_date = plot_daily_forecast(df.copy(), forecast_end_date)

        if forecast_df is not None and last_date is not None:
            display_dashboard(forecast_df, last_date, forecast_end_date)

            # Option to download the full forecast numbers as CSV
            try:
                csv_data = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Forecast CSV",
                    data=csv_data,
                    file_name=f'neuralprophet_forecast_{forecast_end_date.date()}.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Failed to generate download file: {e}")

    # Footer link
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()
