import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
from datetime import timedelta
import logging
import torch # Import torch
from torch import serialization # Import serialization module
from torch.nn import SmoothL1Loss # PyTorch loss class
from torch.optim import AdamW # PyTorch optimizer class
from torch.optim.lr_scheduler import OneCycleLR # PyTorch LR scheduler class

# Import the configuration classes mentioned in previous errors
from neuralprophet.configure import ConfigSeasonality, Season, Train, Trend # <-- ADDED Trend
# Import the loss function class mentioned in the latest error
from neuralprophet.custom_loss_metrics import PinballLoss

# --- MOVE st.set_page_config() HERE ---
# Must be the first Streamlit command executed
st.set_page_config(layout="wide", page_title="GA4 Forecaster (NeuralProphet)")

# Optional: Suppress excessive logging from NeuralProphet during training
set_log_level("ERROR")
# logging.getLogger("NP").setLevel(logging.ERROR) # Alternative way


# --- Add this section to allowlist necessary classes for torch.load ---
# This addresses the "Weights only load failed" error with PyTorch 2.6+.
ADD_SAFE_GLOBALS_MESSAGE = "" # Store message instead of printing directly
try:
    # Add classes that might be pickled/unpickled by NeuralProphet internally.
    safe_globals_list = [
        ConfigSeasonality, Season, Train, Trend, # <-- ADDED Trend
        PinballLoss,                       # NeuralProphet custom loss class
        SmoothL1Loss,                      # PyTorch loss class
        AdamW,                             # PyTorch optimizer class
        OneCycleLR                         # PyTorch LR scheduler class
    ]
    serialization.add_safe_globals(safe_globals_list)
    # Store message to show later inside main()
    ADD_SAFE_GLOBALS_MESSAGE = f"Info: Added {len(safe_globals_list)} class(es) to torch safe globals for compatibility."

except AttributeError:
    ADD_SAFE_GLOBALS_MESSAGE = "Info: torch.serialization.add_safe_globals not used (likely older PyTorch version)."
except ImportError:
    # Make the ImportError message slightly more specific if possible
    ADD_SAFE_GLOBALS_MESSAGE = "Warning: Could not import one or more necessary classes for torch compatibility."
except Exception as e:
    ADD_SAFE_GLOBALS_MESSAGE = f"Warning: An unexpected error occurred while adding safe globals for torch: {e}"
# --- End of allowlist section ---


def load_data():
    """Loads GA4 data from an uploaded CSV file."""
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Basic validation for required columns
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("CSV must contain 'Date' and 'Sessions' columns.")
                return None
            # Additional check for empty dataframe after load
            if df.empty:
                 st.error("Uploaded CSV file is empty.")
                 return None
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    else:
        st.info("Awaiting CSV file upload...")
        return None

def plot_daily_forecast(df, forecast_end_date):
    """
    Fits a NeuralProphet model and plots the actual vs forecasted sessions.

    Args:
        df (pd.DataFrame): DataFrame with historical data ('Date', 'Sessions').
        forecast_end_date (pd.Timestamp): The date to forecast up to.

    Returns:
        tuple: (forecast_df, last_date) or (None, None) on error.
    """
    try:
        # Convert 'Date' column from string format (YYYYMMDD) to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    except ValueError:
        st.error("Error parsing 'Date' column. Ensure it's in YYYYMMDD format.")
        return None, None
    except KeyError:
         st.error("CSV file must contain a 'Date' column.") # Added check
         return None, None
    except Exception as e:
         st.error(f"An error occurred during date conversion: {e}")
         return None, None

    # Ensure 'Sessions' column exists before renaming
    if 'Sessions' not in df.columns:
        st.error("CSV file must contain a 'Sessions' column.")
        return None, None

    # Sort by date just in case
    df = df.sort_values('Date')

    # Rename columns for NeuralProphet ('ds' for date, 'y' for target)
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)

    # Convert 'y' column to numeric, coercing errors (like non-numeric values) to NaN
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Drop rows where 'ds' or 'y' might be NaN (e.g., after coercion or if originally missing)
    initial_rows = len(df)
    df.dropna(subset=['ds', 'y'], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        st.warning(f"Dropped {dropped_rows} rows with missing dates or non-numeric session values.")

    if df.empty:
        st.error("No valid data remaining after processing (check for missing dates or non-numeric sessions).")
        return None, None

    last_date = df['ds'].max()

    # Calculate forecast periods as the number of days from the last observed date
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date in the data.")
        # Provide more context
        st.error(f"Last observed date: {last_date.date()}. Selected end date: {forecast_end_date.date()}")
        return None, last_date

    # --- NeuralProphet Section ---
    # Initialize the NeuralProphet model
    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False, # Usually not needed/helpful for daily website traffic
        # n_lags=7, # Example: Uncomment to add Autoregression (might need more data)
        quantiles=[0.05, 0.95] # For 90% uncertainty intervals
    )

    st.info("Training NeuralProphet model... this may take a few moments.")
    progress_bar = st.progress(0)
    forecast = None # Initialize forecast variable

    try:
        # Check for minimum data points for reliable seasonality/fitting
        min_data_points = 30 # Increased minimum recommendation
        if hasattr(model, 'n_lags') and model.n_lags > 0:
             min_data_points = max(min_data_points, model.n_lags * 2 + 14) # Heuristic if lags are used

        if len(df) < min_data_points:
             st.warning(f"Dataset has only {len(df)} data points (recommended: {min_data_points}+). "
                        "Model results might be less reliable. Consider using more historical data.")
             # Use all data for training if too small to split reliably
             df_train = df
             metrics = model.fit(df_train, freq="D")
             progress_bar.progress(50)

        else:
            # Split data for metrics, fit model
            df_train, df_val = model.split_df(df, freq="D", valid_p=0.1) # Use last 10% for validation
            st.write(f"Training on {len(df_train)} data points, validating on {len(df_val)}.")
            metrics = model.fit(df_train, freq="D") # Fit on training data only
            progress_bar.progress(50) # Mid-point after fitting
            # Optional: Evaluate on validation set and display
            try:
                val_metrics = model.test(df_val)
                st.write("Model Validation Metrics (on held-out 10% data):")
                st.dataframe(val_metrics)
            except Exception as test_e:
                st.warning(f"Could not compute validation metrics: {test_e}")


        st.write("Model Training Metrics (on training data):")
        st.dataframe(metrics) # Display training metrics

        # Create future dataframe & predict
        future = model.make_future_dataframe(df=df, periods=periods) # Pass original df for continuity
        progress_bar.progress(75)
        forecast = model.predict(future)
        progress_bar.progress(100) # Mark as complete

    except Exception as e:
        st.error(f"Error during NeuralProphet fitting or prediction: {e}")
        # Print detailed traceback to console for debugging if running locally
        import traceback
        traceback.print_exc()
        progress_bar.progress(100) # Ensure progress bar finishes on error
        return None, last_date
    # --- End NeuralProphet Section ---


    # Check if forecast is empty or missing essential columns
    if forecast is None or forecast.empty or 'ds' not in forecast.columns or 'yhat1' not in forecast.columns:
         st.error("Prediction failed or returned unexpected results.")
         return None, last_date

    # --- Plotting Section ---
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot actual data
    ax.plot(df['ds'], df['y'], label='Actual', color='blue', marker='.', markersize=4, linestyle='-')

    # Plot forecast line using 'yhat1'
    ax.plot(forecast['ds'], forecast['yhat1'], label='Forecast (yhat1)', color='green', linestyle='--')

    # Plot uncertainty intervals using quantile columns
    # Dynamically get quantile column names based on model's quantiles attribute
    lower_q_col = f'yhat1 {model.quantiles[0]*100:.1f}%'
    upper_q_col = f'yhat1 {model.quantiles[1]*100:.1f}%'

    if lower_q_col in forecast.columns and upper_q_col in forecast.columns:
        ax.fill_between(forecast['ds'],
                        forecast[lower_q_col],
                        forecast[upper_q_col],
                        color='green', alpha=0.2, label=f'Uncertainty Interval ({int(model.quantiles[1]*100-model.quantiles[0]*100)}%)')
    else:
        st.warning(f"Could not find uncertainty columns: '{lower_q_col}', '{upper_q_col}'. Plotting without intervals.")

    # --- Google Update Shading (No changes needed here) ---
    google_updates = [
        ('20230315', '20230328', 'Mar 23 Core'), ('20230822', '20230907', 'Aug 23 Core'),
        ('20230914', '20230928', 'Sep 23 Helpful'), ('20231004', '20231019', 'Oct 23 Core+Spam'),
        ('20231102', '20231204', 'Nov 23 Core+Spam'), ('20240305', '20240419', 'Mar 24 Core'),
        ('20240506', '20240507', 'Site Rep Abuse'), ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'Jun 24 Core'), ('20240815', '20240903', 'Aug 24 Core'),
        # ('20241111', '20241205', 'Nov 24 Core'), # Future dates commented out
        # ('20241212', '20241218', 'Dec 24 Core'),
        # ('20241219', '20241226', 'Dec 24 Spam'),
        # ('20250313', '20250327', 'Mar 25 Core')
    ]
    plot_bottom, plot_top = ax.get_ylim() # Get current y-limits *before* adding text
    text_y_pos = plot_top * 1.02 # Position text slightly above current max y

    for start_str, end_str, label in google_updates:
        try:
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            # Only plot if the update range overlaps with the plotted date range
            if start_date <= forecast['ds'].max() and end_date >= df['ds'].min():
                ax.axvspan(start_date, end_date, color='lightcoral', alpha=0.2) # Changed color slightly
                mid_date = start_date + (end_date - start_date) / 2
                # Plot text only if it falls within the x-axis limits
                if mid_date >= ax.get_xlim()[0] and mid_date <= ax.get_xlim()[1]:
                     ax.text(mid_date, text_y_pos, label, ha='center', va='bottom', fontsize=8, rotation=90, color='dimgray')
        except Exception as e:
            st.warning(f"Could not plot Google Update '{label}': {e}")

    # Reset y-limits slightly expanded to ensure text fits if it was plotted
    ax.set_ylim(bottom=plot_bottom, top=text_y_pos * 1.05) # Add padding above text

    ax.set_title('Daily Actual vs. Forecasted GA4 Sessions (NeuralProphet) with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6) # Add grid for readability
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    st.pyplot(fig)

    return forecast, last_date


def display_dashboard(forecast, last_date, forecast_end_date):
    """Displays the forecast data table and summary metrics."""
    st.subheader("Forecast Data Table")

    # Dynamically get quantile column names based on model's actual quantiles
    lower_q_col = f'yhat1 {forecast.columns[forecast.columns.str.contains("%")][0].split(" ")[-1]}' if any('%' in col and 'yhat1' in col for col in forecast.columns) else 'yhat1 5.0%'
    upper_q_col = f'yhat1 {forecast.columns[forecast.columns.str.contains("%")][-1].split(" ")[-1]}' if any('%' in col and 'yhat1' in col for col in forecast.columns) else 'yhat1 95.0%'


    # Check if quantile columns exist, handle gracefully if not
    if lower_q_col not in forecast.columns:
        forecast[lower_q_col] = pd.NA # Use pandas NA for consistency
        st.warning(f"Column '{lower_q_col}' not found in forecast data.")
    if upper_q_col not in forecast.columns:
        forecast[upper_q_col] = pd.NA
        st.warning(f"Column '{upper_q_col}' not found in forecast data.")


    # Filter forecast for the future period
    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)].copy() # Use copy to avoid SettingWithCopyWarning

    # Select and rename columns for display
    display_cols = ['ds', 'yhat1', lower_q_col, upper_q_col]
    # Ensure all display columns exist before selecting
    display_cols = [col for col in display_cols if col in forecast_filtered.columns]
    forecast_display = forecast_filtered[display_cols]

    # Rename after selection
    rename_map = {
        'yhat1': 'Forecast',
        lower_q_col: f'Lower Bound ({lower_q_col.split(" ")[-1]})' if lower_q_col in forecast_display else 'Lower Bound',
        upper_q_col: f'Upper Bound ({upper_q_col.split(" ")[-1]})' if upper_q_col in forecast_display else 'Upper Bound'
    }
    forecast_display = forecast_display.rename(columns=rename_map)

    # Format date column and numbers for better readability
    forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
    # Format numeric columns (handle potential NAs)
    for col in forecast_display.columns:
        if col != 'ds' and pd.api.types.is_numeric_dtype(forecast_display[col]):
             forecast_display[col] = forecast_display[col].map('{:,.0f}'.format, na_action='ignore') # Format as integer with comma

    st.dataframe(forecast_display, use_container_width=True) # Use full width

    # --- Forecast Summary ---
    st.subheader("Forecast Summary")
    horizon = (forecast_end_date - last_date).days
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon} days")

    # Get the forecast row closest to the forecast end date
    forecast_future = forecast[forecast['ds'] > last_date]
    if forecast_future.empty:
        st.write("No forecast data available for the selected future date range.")
        return # Exit early if no future data

    closest_idx = (forecast_future['ds'] - forecast_end_date).abs().idxmin()
    forecast_value = forecast_future.loc[closest_idx]

    # Calculate delta for metric based on quantiles if available and valid
    delta_val = pd.NA
    delta_str = "Range N/A"
    if pd.notna(forecast_value.get(lower_q_col)) and pd.notna(forecast_value.get(upper_q_col)):
         delta_val = forecast_value[upper_q_col] - forecast_value[lower_q_col]
         # Check if delta_val is not NaN/NA before formatting
         if pd.notna(delta_val):
              delta_str = f"Range: {int(delta_val):,}" # Add comma formatting

    # Handle potential NA in forecast value for metric
    metric_value = int(forecast_value['yhat1']) if pd.notna(forecast_value['yhat1']) else "N/A"
    if isinstance(metric_value, int):
         metric_value_display = f"{metric_value:,}" # Add comma formatting
    else:
         metric_value_display = metric_value


    st.metric(label=f"Forecasted Traffic (at {forecast_end_date.date()})",
              value=metric_value_display,
              delta=delta_str)

    # --- Year-over-Year Calculation ---
    st.subheader("Year-over-Year Comparison (Forecast vs Forecast)")
    start_forecast = last_date + pd.Timedelta(days=1)
    end_forecast = forecast_end_date
    current_period = forecast[(forecast['ds'] >= start_forecast) & (forecast['ds'] <= end_forecast)]

    # Define the corresponding period one year earlier using forecast data
    start_prev = start_forecast - pd.Timedelta(days=365)
    end_prev = end_forecast - pd.Timedelta(days=365)
    prev_period = forecast[(forecast['ds'] >= start_prev) & (forecast['ds'] <= end_prev)]

    if not current_period.empty and not prev_period.empty:
        # Ensure lengths match for a fair comparison (handle potential partial year data)
        if len(current_period) != len(prev_period):
            st.warning(f"YoY Comparison Warning: Periods have different lengths ({len(current_period)} vs {len(prev_period)} days). Comparison might be skewed.")
            # Optional: Adjust periods to match length if desired, but summing available data is simpler

        current_sum = current_period['yhat1'].sum()
        prev_sum = prev_period['yhat1'].sum()

        change_label = "N/A"
        if pd.notna(current_sum) and pd.notna(prev_sum):
            if prev_sum != 0:
                yoy_change = ((current_sum - prev_sum) / prev_sum) * 100
                change_label = f"{yoy_change:.2f}%"
            elif current_sum > 0:
                 change_label = "inf%" # Previous was zero, current is positive
            else:
                 change_label = "0.00%" # Both are zero

        st.write(f"Total Forecasted ({start_forecast.date()} to {end_forecast.date()}): {current_sum:,.0f}")
        st.write(f"Total Forecasted ({start_prev.date()} to {end_prev.date()}): {prev_sum:,.0f}")
        st.write(f"Year-over-Year Change: {change_label}")
    else:
        st.warning("Not enough historical forecast data within the current run for Year-over-Year calculation.")
        st.write(f"(Requires forecast data covering {start_prev.date()} to {end_prev.date()})")


def main():
    """Main function to run the Streamlit app."""
    # --- Page Config moved to top ---

    # Display the stored message from the add_safe_globals block
    if ADD_SAFE_GLOBALS_MESSAGE:
        if "Warning" in ADD_SAFE_GLOBALS_MESSAGE:
            st.warning(ADD_SAFE_GLOBALS_MESSAGE)
        else:
            st.info(ADD_SAFE_GLOBALS_MESSAGE) # Use info for non-warning messages

    st.title("📊 GA4 Daily Forecasting with NeuralProphet")
    st.write("""
        Upload your Google Analytics 4 daily sessions data (CSV) to generate a forecast using NeuralProphet.
        The app visualizes historical data, the forecast, uncertainty intervals, and relevant Google Algorithm Updates.
    """)
    st.markdown("""
        **CSV Requirements:**
        - Must contain columns named "**Date**" (format YYYYMMDD) and "**Sessions**".
        - Data should ideally be sorted chronologically (oldest first).
        - Ensure 'Sessions' column contains only numeric values.
    """)

    # --- Sidebar for Inputs ---
    st.sidebar.header("Configuration")
    # Set default end date and allow user selection
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    min_forecast_date = (pd.Timestamp.today() + timedelta(days=1)).date() # Forecast must be in future
    forecast_end_date_input = st.sidebar.date_input(
        "Select Forecast End Date",
        value=default_forecast_end,
        min_value=min_forecast_date,
        help="Choose the date up to which you want to forecast."
    )
    forecast_end_date = pd.to_datetime(forecast_end_date_input)

    st.sidebar.markdown("---")
    st.sidebar.info("Ensure `neuralprophet`, `torch`, `pandas`, `matplotlib`, and `streamlit` are installed.")
    # --- End Sidebar ---


    # --- Main Area ---
    # Load GA4 data
    df_original = load_data()

    if df_original is not None:
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df_original.head(), use_container_width=True)

        # Perform forecasting and plotting
        # Pass a copy of df to avoid modifying the original previewed df
        forecast_df, last_date = plot_daily_forecast(df_original.copy(), forecast_end_date)

        # Display dashboard and download button if forecast was successful
        if forecast_df is not None and last_date is not None:
            display_dashboard(forecast_df, last_date, forecast_end_date)

            # Option to download the full forecast numbers as CSV
            try:
                # Select relevant columns for download
                download_cols = [col for col in forecast_df.columns if 'yhat' in col or col == 'ds' or '%' in col]
                csv_data = forecast_df[download_cols].to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8')
                st.download_button(
                    label="💾 Download Full Forecast CSV",
                    data=csv_data,
                    file_name=f'neuralprophet_forecast_{forecast_end_date.date()}.csv',
                    mime='text/csv',
                    help="Downloads the forecast values and uncertainty bounds."
                )
            except Exception as e:
                st.error(f"Failed to generate download file: {e}")
        else:
            st.error("Forecasting could not be completed due to previous errors.") # Shows if plot_daily_forecast returned None

    # Footer link
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")
    # --- End Main Area ---

if __name__ == "__main__":
    main()
