import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For date conversions in plotting
from neuralprophet import NeuralProphet, set_log_level
from datetime import timedelta, date # Import date
import logging
import traceback # For printing tracebacks

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

# --- Updated plot_daily_forecast Function ---
def plot_daily_forecast(df_processed, forecast_end_date, baseline_date=None):
    """
    Fits NeuralProphet model(s) and plots actuals vs forecasts.
    Optionally includes a baseline forecast comparison.

    Args:
        df_processed (pd.DataFrame): Cleaned historical data.
        forecast_end_date (pd.Timestamp): Final date for future forecast.
        baseline_date (pd.Timestamp, optional): Date to train baseline model up to.
                                                 Defaults to None (no baseline plot).
    """
    if df_processed is None or df_processed.empty:
        st.error("Cannot plot forecast: No valid processed data available.")
        return None, None, None # Return None for forecast, baseline, last_date

    last_date = df_processed['ds'].max()
    forecast_baseline = None # Initialize baseline forecast

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(16, 8))
    # Plot ALL actual data first
    ax.plot(df_processed['ds'], df_processed['y'], label='Actual Sessions', color='blue', marker='.', markersize=4, linestyle='-')


    # --- Baseline Forecast (Optional) ---
    if baseline_date and baseline_date < last_date:
        st.info(f"Calculating baseline forecast (trained up to {baseline_date.date()})...")
        df_train_baseline = df_processed[df_processed['ds'] <= baseline_date]

        if len(df_train_baseline) < 30: # Need sufficient data for baseline
             st.warning(f"Not enough data (found {len(df_train_baseline)}) before {baseline_date.date()} to create reliable baseline.")
        else:
            try:
                model_baseline = NeuralProphet(
                    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    # No quantiles needed for baseline plot, simpler model
                )
                # Fit baseline model ONLY on data up to baseline_date
                metrics_baseline = model_baseline.fit(df_train_baseline, freq="D")
                st.write(f"Baseline Model Training Metrics (up to {baseline_date.date()}):")
                st.dataframe(metrics_baseline.tail(1)) # Show final metrics

                # Predict from baseline_date up to last_date of actuals
                periods_baseline = (last_date - baseline_date).days
                future_baseline = model_baseline.make_future_dataframe(df_train_baseline, periods=periods_baseline)
                forecast_baseline = model_baseline.predict(future_baseline)

                # Plot the baseline forecast (only after baseline_date)
                baseline_plot_data = forecast_baseline[forecast_baseline['ds'] > baseline_date]
                ax.plot(baseline_plot_data['ds'], baseline_plot_data['yhat1'],
                        label=f'Baseline Forecast (from {baseline_date.date()})',
                        color='darkorange', linestyle=':') # Distinct style

            except Exception as e_base:
                st.error(f"Error creating baseline forecast: {e_base}")
                st.error(traceback.format_exc())
                forecast_baseline = None # Ensure it's None on error
    elif baseline_date:
         st.warning(f"Baseline date {baseline_date.date()} is not before the last data date {last_date.date()}. Cannot create baseline comparison.")

    # --- Future Forecast (Based on ALL data) ---
    st.info("Calculating future forecast (trained on all available data)...")
    periods_future = (forecast_end_date - last_date).days
    if periods_future <= 0:
        st.warning("Selected forecast end date is not after the last data date. No future forecast generated.")
        forecast_future = None
    else:
        try:
            model_future = NeuralProphet(
                yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                quantiles=[0.05, 0.95] # Include quantiles for future uncertainty
            )
            # Fit future model on ALL processed data
            metrics_future = model_future.fit(df_processed, freq="D")
            st.write("Future Forecast Model Training Metrics (All Data):")
            st.dataframe(metrics_future.tail(1)) # Show final metrics

            # Predict only the future period
            future_df = model_future.make_future_dataframe(df_processed, periods=periods_future)
            forecast_future = model_future.predict(future_df)

            # Plot only the future part of this forecast
            future_plot_data = forecast_future[forecast_future['ds'] > last_date]
            ax.plot(future_plot_data['ds'], future_plot_data['yhat1'],
                    label='Future Forecast', color='green', linestyle='--')

            # Plot uncertainty intervals for the FUTURE forecast
            lower_q_col = 'yhat1 5.0%'
            upper_q_col = 'yhat1 95.0%'
            uncertainty_label = 'Future Uncertainty (90%)'
            if lower_q_col in future_plot_data.columns and upper_q_col in future_plot_data.columns:
                ax.fill_between(future_plot_data['ds'],
                                future_plot_data[lower_q_col],
                                future_plot_data[upper_q_col],
                                color='green', alpha=0.2, label=uncertainty_label)
            else:
                st.warning("Could not find future uncertainty columns.")

        except Exception as e_future:
            st.error(f"Error creating future forecast: {e_future}")
            st.error(traceback.format_exc())
            forecast_future = None

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
    xlim_min, xlim_max = ax.get_xlim() # Get final limits after plotting data

    for start_str, end_str, label in google_updates:
        try:
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            # Check against overall data range
            if start_date <= forecast_end_date and end_date >= df_processed['ds'].min():
                ax.axvspan(start_date, end_date, color='lightcoral', alpha=0.2)
                mid_date = start_date + (end_date - start_date) / 2
                mid_date_num = mdates.date2num(mid_date)
                # Check if midpoint is within the final plotted x-limits
                if mid_date_num >= xlim_min and mid_date_num <= xlim_max:
                     ax.text(mid_date, text_y_pos, label, ha='center', va='bottom', fontsize=8, rotation=90, color='dimgray')
        except Exception as e:
            st.warning(f"Could not plot Google Update '{label}': {e}")

    ax.set_ylim(bottom=max(0, plot_bottom * 0.95), top=text_y_pos * 1.05) # Ensure y starts >=0
    ax.set_title('GA4 Sessions: Actual vs. Forecasts with Google Updates')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

    # Return the FUTURE forecast (for dashboard) and the baseline (if created)
    return forecast_future, forecast_baseline, last_date


# --- Updated display_dashboard Function ---
# It now only needs the FUTURE forecast for metrics and table
def display_dashboard(forecast_future, last_date, forecast_end_date, df_processed):
    """Displays the future forecast data table and summary metrics."""

    if forecast_future is None:
         st.warning("Future forecast data not available for dashboard.")
         # Still attempt YoY calculation below if possible
    else:
        st.subheader("Future Forecast Data Table")
        lower_q_col = 'yhat1 5.0%'
        upper_q_col = 'yhat1 95.0%'

        if lower_q_col not in forecast_future.columns: forecast_future[lower_q_col] = pd.NA
        if upper_q_col not in forecast_future.columns: forecast_future[upper_q_col] = pd.NA

        # Filter only future dates for the table
        forecast_filtered = forecast_future[forecast_future['ds'] > last_date].copy()

        display_cols = ['ds', 'yhat1', lower_q_col, upper_q_col]
        display_cols = [col for col in display_cols if col in forecast_filtered.columns]
        forecast_display = forecast_filtered[display_cols]

        rename_map = { 'yhat1': 'Forecast',
            lower_q_col: f'Lower Bound ({lower_q_col.split(" ")[-1]})' if lower_q_col in forecast_display else 'Lower Bound',
            upper_q_col: f'Upper Bound ({upper_q_col.split(" ")[-1]})' if upper_q_col in forecast_display else 'Upper Bound'}
        forecast_display = forecast_display.rename(columns=rename_map)

        forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
        for col in forecast_display.columns:
            if col != 'ds' and pd.api.types.is_numeric_dtype(forecast_display[col]):
                 forecast_display[col] = forecast_display[col].map('{:,.0f}'.format, na_action='ignore')

        st.dataframe(forecast_display, use_container_width=True)

        # --- Forecast Summary (based on future forecast) ---
        st.subheader("Future Forecast Summary")
        horizon = (forecast_end_date - last_date).days
        st.write(f"Forecast End Date: {forecast_end_date.date()}")
        st.write(f"Forecast Horizon: {horizon} days")

        forecast_future_only = forecast_future[forecast_future['ds'] > last_date]
        if not forecast_future_only.empty:
            closest_idx = (forecast_future_only['ds'] - forecast_end_date).abs().idxmin()
            forecast_value = forecast_future_only.loc[closest_idx]
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


    # --- Year-over-Year Calculation (uses future forecast vs actual) ---
    st.subheader("Year-over-Year Comparison (Future Forecast vs. Past Actual)")
    start_forecast = last_date + pd.Timedelta(days=1)
    end_forecast = forecast_end_date

    if forecast_future is None:
         current_period = pd.DataFrame() # Empty df if no future forecast
    else:
         # Current period uses FUTURE FORECAST data
         current_period = forecast_future[(forecast_future['ds'] >= start_forecast) & (forecast_future['ds'] <= end_forecast)]

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
    # ... (rest of title/description) ...
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

    # --- New Baseline Date Input ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Baseline Comparison (Optional)")
    enable_baseline = st.sidebar.checkbox("Compare actuals to historical baseline?", value=False)
    baseline_date_input = None
    if enable_baseline:
         # Suggest a default before the first major 2023 update
         default_baseline_date = date(2023, 3, 14)
         baseline_date_input = st.sidebar.date_input(
             "Train Baseline Model Up To:",
             value=default_baseline_date,
             # Min/max values can be set dynamically later if needed based on data
             help="Model trained only on data BEFORE this date to show a baseline forecast."
         )

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
        # Convert baseline input date to Timestamp if enabled
        baseline_comparison_date = pd.to_datetime(baseline_date_input) if enable_baseline and baseline_date_input else None

        # Perform forecasting and plotting
        # Pass baseline_comparison_date to the plotting function
        forecast_future_df, forecast_baseline_df, _ = plot_daily_forecast(
            df_processed.copy(), forecast_end_date, baseline_comparison_date
        )

        # Display dashboard (using future forecast and processed actuals)
        display_dashboard(forecast_future_df, last_date, forecast_end_date, df_processed)

        # Option to download the FUTURE forecast numbers
        if forecast_future_df is not None:
            try:
                download_cols = [col for col in forecast_future_df.columns if 'yhat' in col or col == 'ds' or '%' in col]
                csv_data = forecast_future_df[download_cols].to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8')
                st.download_button(
                    label="💾 Download Future Forecast CSV", data=csv_data, # Clarified label
                    file_name=f'neuralprophet_future_forecast_{forecast_end_date.date()}.csv', mime='text/csv',
                    help="Downloads the future forecast values and uncertainty bounds." )
            except Exception as e:
                st.error(f"Failed to generate download file: {e}")
        # Optionally add download for baseline forecast if needed
        # if forecast_baseline_df is not None: ... download baseline ...

    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()
