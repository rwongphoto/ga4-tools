mport streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For date conversions in plotting
from neuralprophet import NeuralProphet, set_log_level
from datetime import timedelta
import logging
import traceback # For printing tracebacks

# NOTE: Safe globals imports are likely unnecessary if using older PyTorch (<2.6)
# Comment them out if you downgraded PyTorch successfully.
# import torch
# from torch import serialization
# ... other imports for safe_globals ...

# --- Page Config (First Streamlit Command) ---
st.set_page_config(layout="wide", page_title="GA4 Forecaster (NeuralProphet)")

# Optional: Suppress excessive logging
set_log_level("ERROR")

# --- Allowlist section (likely inactive with older PyTorch) ---
ADD_SAFE_GLOBALS_MESSAGE = "Info: Using older PyTorch version, safe_globals allowlisting may not be active."
# ...(rest of commented out or active safe_globals block)...


def load_data():
    """Loads GA4 data from an uploaded CSV file."""
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                 st.error("Uploaded CSV file is empty.")
                 return None
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("CSV must contain 'Date' and 'Sessions' columns.")
                return None
            try: import numpy
            except ImportError:
                st.error("NumPy library is required. `pip install numpy`")
                return None
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    else:
        st.info("Awaiting CSV file upload...")
        return None

def clean_data(df):
    """Performs initial data cleaning and validation."""
    if df is None: return None, "No DataFrame provided."
    df_clean = df.copy()
    try: df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%Y%m%d')
    except ValueError: return None, "Error parsing 'Date'. Use YYYYMMDD format."
    except KeyError: return None, "Missing 'Date' column."
    except Exception as e: return None, f"Date conversion error: {e}"
    if 'Sessions' not in df_clean.columns: return None, "Missing 'Sessions' column."
    df_clean = df_clean.sort_values('Date')
    df_clean.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    df_clean['y'] = pd.to_numeric(df_clean['y'], errors='coerce')
    initial_rows = len(df_clean)
    df_clean.dropna(subset=['ds', 'y'], inplace=True)
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0: st.warning(f"Dropped {dropped_rows} rows with invalid data.")
    if df_clean.empty: return None, "No valid data remaining after cleaning."
    return df_clean, None

# --- Simplified plot_daily_forecast Function ---
def plot_daily_forecast(df_processed, forecast_end_date):
    """
    Fits a single NeuralProphet model on all data and plots
    actuals vs. model fit and future forecast.

    Args:
        df_processed (pd.DataFrame): Cleaned historical data.
        forecast_end_date (pd.Timestamp): Final date for future forecast.

    Returns:
        tuple: (forecast_df, last_date) or (None, None) on error.
               forecast_df contains both historical fit and future predictions.
    """
    if df_processed is None or df_processed.empty:
        st.error("Cannot plot forecast: No valid processed data available.")
        return None, None

    last_date = df_processed['ds'].max()
    periods_future = (forecast_end_date - last_date).days

    if periods_future <= 0:
        st.warning("Selected forecast end date is not after the last data date. Only historical fit will be shown.")
        # Allow proceeding to show the fit, but set periods_future to 0 for make_future_dataframe
        periods_future = 0
        # Alternatively, return an error if no future forecast is desired:
        # st.error("Forecast end date must be after the last observed date.")
        # return None, last_date

    # --- Single Model Fit (on ALL data) ---
    st.info("Training model and forecasting (trained on all available data)...")
    model = NeuralProphet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        quantiles=[0.05, 0.95] # Include quantiles for uncertainty
    )

    progress_bar = st.progress(0)
    forecast = None # Initialize forecast

    try:
        min_data_points = 30
        # (Validation split logic remains useful for monitoring fit)
        if len(df_processed) < min_data_points:
             st.warning(f"Dataset has only {len(df_processed)} valid data points (recommended: {min_data_points}+). Results might be less reliable.")
             df_train = df_processed
             metrics = model.fit(df_train, freq="D")
             progress_bar.progress(50)
        else:
            df_train, df_val = model.split_df(df_processed, freq="D", valid_p=0.1)
            st.write(f"Training on {len(df_train)} data points, validating on {len(df_val)}.")
            metrics = model.fit(df_train, freq="D")
            progress_bar.progress(50)
            try:
                val_metrics = model.test(df_val)
                st.write("Model Validation Metrics (on held-out 10% data):")
                st.dataframe(val_metrics)
            except Exception as test_e:
                st.warning(f"Could not compute validation metrics: {test_e}")

        st.write("Model Training Metrics (on training data):")
        st.dataframe(metrics)

        # Create future dataframe (includes history + future)
        future_df = model.make_future_dataframe(df=df_processed, periods=periods_future)
        progress_bar.progress(75)
        # Predict history + future
        forecast = model.predict(future_df)
        progress_bar.progress(100)

    except Exception as e:
        st.error(f"Error during NeuralProphet fitting or prediction: {e}")
        st.error(traceback.format_exc()) # Show full traceback
        progress_bar.progress(100)
        return None, last_date
    # --- End NeuralProphet Section ---

    if forecast is None or forecast.empty or 'ds' not in forecast.columns or 'yhat1' not in forecast.columns:
         st.error("Prediction failed or returned unexpected results.")
         return None, last_date

    # --- Plotting Section ---
    fig, ax = plt.subplots(figsize=(16, 8))
    # Plot actual using the processed data
    ax.plot(df_processed['ds'], df_processed['y'], label='Actual Sessions', color='blue', marker='.', markersize=4, linestyle='-')
    # Plot model fit + forecast (single green line)
    ax.plot(forecast['ds'], forecast['yhat1'], label='Model Fit / Forecast', color='green', linestyle='--') # Updated label

    # Plot uncertainty intervals (covers fit + forecast period)
    lower_q_col = 'yhat1 5.0%'
    upper_q_col = 'yhat1 95.0%'
    uncertainty_label = 'Uncertainty Interval (90%)'

    if lower_q_col in forecast.columns and upper_q_col in forecast.columns:
        ax.fill_between(forecast['ds'],
                        forecast[lower_q_col],
                        forecast[upper_q_col],
                        color='green', alpha=0.2, label=uncertainty_label)
    else:
        st.warning(f"Could not find uncertainty columns: '{lower_q_col}', '{upper_q_col}'.")

    # --- Google Update Shading ---
    google_updates = [
        ('20230315', '20230328', 'Mar 23 Core'), ('20230822', '20230907', 'Aug 23 Core'),
        ('20230914', '20230928', 'Sep 23 Helpful'), ('20231004', '20231019', 'Oct 23 Core+Spam'),
        ('20231102', '20231204', 'Nov 23 Core+Spam'), ('20240305', '20240419', 'Mar 24 Core'),
        ('20240506', '20240507', 'Site Rep Abuse'), ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'Jun 24 Core'), ('20240815', '20240903', 'Aug 24 Core'),
        ('20241111', '20241205', 'Nov 24 Core'), ('20241212', '20241218', 'Dec 24 Core'),
        ('20241219', '20241226', 'Dec 24 Spam'), ('20250313', '20250327', 'Mar 25 Core')
    ]
    plot_bottom, plot_top = ax.get_ylim()
    text_y_pos = plot_top * 1.02
    # Get limits *after* plotting lines
    xlim_min, xlim_max = ax.get_xlim()

    for start_str, end_str, label in google_updates:
        try:
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            # Use forecast max date which includes future
            if start_date <= forecast['ds'].max() and end_date >= df_processed['ds'].min():
                ax.axvspan(start_date, end_date, color='lightcoral', alpha=0.2)
                mid_date = start_date + (end_date - start_date) / 2
                mid_date_num = mdates.date2num(mid_date)
                if mid_date_num >= xlim_min and mid_date_num <= xlim_max:
                     ax.text(mid_date, text_y_pos, label, ha='center', va='bottom', fontsize=8, rotation=90, color='dimgray')
        except Exception as e:
            st.warning(f"Could not plot Google Update '{label}': {e}")

    ax.set_ylim(bottom=max(0, plot_bottom * 0.95), top=text_y_pos * 1.05)
    ax.set_title('GA4 Sessions: Actual vs. Model Fit & Forecast with Google Updates') # Updated title
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

    # Return the combined forecast (fit+future) and last_date
    return forecast, last_date

# --- display_dashboard remains largely the same, but uses combined forecast ---
def display_dashboard(forecast, last_date, forecast_end_date, df_processed):
    """Displays the forecast data table and summary metrics."""
    if forecast is None or df_processed is None:
         st.warning("Cannot display dashboard, missing forecast or processed data.")
         return

    st.subheader("Forecast Data Table (Future Period)") # Clarified table shows future
    lower_q_col = 'yhat1 5.0%'
    upper_q_col = 'yhat1 95.0%'

    # Create columns if they don't exist (e.g., if quantiles failed)
    if lower_q_col not in forecast.columns: forecast[lower_q_col] = pd.NA
    if upper_q_col not in forecast.columns: forecast[upper_q_col] = pd.NA

    # Filter only future dates for the TABLE display
    forecast_filtered_future = forecast[forecast['ds'] > last_date].copy()

    if forecast_filtered_future.empty:
        st.info("No future dates to display in the table.")
    else:
        display_cols = ['ds', 'yhat1', lower_q_col, upper_q_col]
        display_cols = [col for col in display_cols if col in forecast_filtered_future.columns]
        forecast_display = forecast_filtered_future[display_cols]

        rename_map = { 'yhat1': 'Forecast',
            lower_q_col: f'Lower Bound ({lower_q_col.split(" ")[-1]})',
            upper_q_col: f'Upper Bound ({upper_q_col.split(" ")[-1]})'}
        # Only rename if columns exist
        forecast_display = forecast_display.rename(columns={k:v for k,v in rename_map.items() if k in forecast_display})


        forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
        for col in forecast_display.columns:
            if col != 'ds' and pd.api.types.is_numeric_dtype(forecast_display[col]):
                 forecast_display[col] = forecast_display[col].map('{:,.0f}'.format, na_action='ignore')

        st.dataframe(forecast_display, use_container_width=True)

    # --- Forecast Summary (uses future part of the forecast) ---
    st.subheader("Forecast Summary")
    horizon = (forecast_end_date - last_date).days
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon} days")

    # Use the same future-filtered data for metrics
    if not forecast_filtered_future.empty:
        closest_idx = (forecast_filtered_future['ds'] - forecast_end_date).abs().idxmin()
        forecast_value = forecast_filtered_future.loc[closest_idx]
        delta_val = pd.NA
        delta_str = "Range N/A"
        if pd.notna(forecast_value.get(lower_q_col)) and pd.notna(forecast_value.get(upper_q_col)):
             delta_val = forecast_value[upper_q_col] - forecast_value[lower_q_col]
             if pd.notna(delta_val): delta_str = f"Range: {int(delta_val):,}"
        metric_value = int(forecast_value['yhat1']) if pd.notna(forecast_value['yhat1']) else "N/A"
        metric_value_display = f"{metric_value:,}" if isinstance(metric_value, int) else metric_value
        st.metric(label=f"Forecasted Traffic (at {forecast_end_date.date()})", value=metric_value_display, delta=delta_str)
    else:
        st.write("No future forecast data for summary metric.")


    # --- Year-over-Year Calculation (Future Forecast vs. Past Actual) ---
    # This calculation remains the same logic
    st.subheader("Year-over-Year Comparison (Future Forecast vs. Past Actual)")
    start_forecast = last_date + pd.Timedelta(days=1)
    end_forecast = forecast_end_date
    # Current period uses FUTURE FORECAST data from the combined forecast df
    current_period = forecast[(forecast['ds'] >= start_forecast) & (forecast['ds'] <= end_forecast)]

    # Previous period uses ACTUAL historical data
    start_prev = start_forecast - pd.Timedelta(days=365)
    end_prev = end_forecast - pd.Timedelta(days=365)
    prev_period_actual = df_processed[(df_processed['ds'] >= start_prev) & (df_processed['ds'] <= end_prev)]

    if not current_period.empty and not prev_period_actual.empty:
        current_sum = current_period['yhat1'].sum()
        prev_sum = prev_period_actual['y'].sum()
        if len(current_period) != len(prev_period_actual):
            st.warning(f"YoY Warning: Periods have different lengths ({len(current_period)} vs {len(prev_period_actual)} days).")
        change_label = "N/A"
        if pd.notna(current_sum) and pd.notna(prev_sum):
            if prev_sum != 0: change_label = f"{((current_sum - prev_sum) / prev_sum) * 100:.2f}%"
            elif current_sum > 0: change_label = "inf%"
            else: change_label = "0.00%"
        st.write(f"Total **Forecasted** ({start_forecast.date()} to {end_forecast.date()}): {current_sum:,.0f}")
        st.write(f"Total **Actual** ({start_prev.date()} to {end_prev.date()}): {prev_sum:,.0f}")
        st.write(f"Year-over-Year Change (Forecast vs. Actual): {change_label}")
    else:
        warning_msg = "YoY calculation requires:"
        if current_period.empty: warning_msg += "\n- Future forecast data for the selected period."
        if prev_period_actual.empty: warning_msg += f"\n- Actual historical data covering {start_prev.date()} to {end_prev.date()}."
        st.warning(warning_msg)

# --- Updated main Function ---
def main():
    """Main function to run the Streamlit app."""
    if ADD_SAFE_GLOBALS_MESSAGE:
        st.info(ADD_SAFE_GLOBALS_MESSAGE)

    st.title("📊 GA4 Daily Forecasting with NeuralProphet")
    st.write("""
        Upload your Google Analytics 4 daily sessions data (CSV) to generate a forecast using NeuralProphet.
        The app visualizes historical data, the model's fit to history, the future forecast, uncertainty intervals, and relevant Google Algorithm Updates.
    """) # Updated description
    st.markdown("""
        **CSV Requirements:**
        - Must contain columns named "**Date**" (format YYYYMMDD) and "**Sessions**".
        - Data should ideally be sorted chronologically (oldest first).
        - Ensure 'Sessions' column contains only numeric values.
    """)

    # --- Sidebar for Inputs ---
    st.sidebar.header("Configuration")
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    min_forecast_date = (pd.Timestamp.today() + timedelta(days=1)).date()
    forecast_end_date_input = st.sidebar.date_input(
        "Select Forecast End Date", value=default_forecast_end, min_value=min_forecast_date,
        help="Choose the date up to which you want to forecast.")
    forecast_end_date = pd.to_datetime(forecast_end_date_input)

    # --- Baseline Input REMOVED ---

    st.sidebar.markdown("---")
    st.sidebar.info("Using older PyTorch (<2.6). Ensure libraries are installed.")

    # --- Main Area ---
    df_original = load_data()
    df_processed = None
    last_date = None

    if df_original is not None:
        st.subheader("Data Preview (Raw Upload - First 5 Rows)")
        st.dataframe(df_original.head(), use_container_width=True)

        df_processed, error_msg = clean_data(df_original)
        if error_msg:
            st.error(f"Data Cleaning Error: {error_msg}")
            df_processed = None
        elif df_processed is not None:
             last_date = df_processed['ds'].max()
             st.success(f"Data cleaned successfully. Last date: {last_date.date()}")


    if df_processed is not None and last_date is not None:
        # Perform forecasting and plotting (no baseline date passed)
        forecast_df, _ = plot_daily_forecast(
            df_processed.copy(), forecast_end_date
        ) # Returns the combined fit+future forecast

        # Display dashboard and download button if forecast was successful
        if forecast_df is not None:
            # Pass the combined forecast_df
            display_dashboard(forecast_df, last_date, forecast_end_date, df_processed)
            try:
                # Download includes fit + future prediction
                download_cols = [col for col in forecast_df.columns if 'yhat' in col or col == 'ds' or '%' in col]
                csv_data = forecast_df[download_cols].to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8')
                st.download_button(
                    label="💾 Download Full Forecast CSV (Fit + Future)", # Clarified label
                    data=csv_data,
                    file_name=f'neuralprophet_full_forecast_{forecast_end_date.date()}.csv',
                    mime='text/csv',
                    help="Downloads historical fit, future forecast, and uncertainty bounds." )
            except Exception as e:
                st.error(f"Failed to generate download file: {e}")
        else:
            st.error("Forecasting step failed. Cannot display dashboard.")

    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()

