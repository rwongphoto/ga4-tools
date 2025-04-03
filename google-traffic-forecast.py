import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
from datetime import timedelta
import logging
# NOTE: Since PyTorch was downgraded (e.g., to <2.6), the lengthy
#       'add_safe_globals' section and its imports are likely no longer
#       needed and can be commented out or removed if desired.
#       Keeping them doesn't hurt, but they won't be used by older PyTorch.
# import torch
# from torch import serialization
# from torch.nn import SmoothL1Loss
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import OneCycleLR
# from numpy.core.multiarray import _reconstruct
# from numpy import ndarray, dtype, float64
# from numpy.dtypes import Float64DType
# from neuralprophet.configure import ConfigSeasonality, Season, Train, Trend, AR
# from neuralprophet.custom_loss_metrics import PinballLoss

# --- MOVE st.set_page_config() HERE ---
st.set_page_config(layout="wide", page_title="GA4 Forecaster (NeuralProphet)")

# Optional: Suppress excessive logging from NeuralProphet during training
set_log_level("ERROR")
# logging.getLogger("NP").setLevel(logging.ERROR) # Alternative way

# --- Allowlist section (likely inactive with older PyTorch) ---
ADD_SAFE_GLOBALS_MESSAGE = "Info: Using older PyTorch version, safe_globals allowlisting may not be active."
# try:
#     safe_globals_list = [
#         ConfigSeasonality, Season, Train, Trend, AR, PinballLoss,
#         SmoothL1Loss, AdamW, OneCycleLR, _reconstruct,
#         ndarray, dtype, float64, Float64DType ]
#     # Check if serialization and add_safe_globals exist before calling
#     if 'serialization' in globals() and hasattr(serialization, 'add_safe_globals'):
#          serialization.add_safe_globals(safe_globals_list)
#          ADD_SAFE_GLOBALS_MESSAGE = f"Info: Added {len(safe_globals_list)} items to torch safe globals (may be ignored by older PyTorch)."
# except NameError:
#      # One of the imports failed, likely because torch wasn't imported fully above
#      ADD_SAFE_GLOBALS_MESSAGE = "Warning: Could not perform safe_globals setup (likely due to older PyTorch)."
# except AttributeError:
#     ADD_SAFE_GLOBALS_MESSAGE = "Info: torch.serialization.add_safe_globals not used (older PyTorch version)."
# except ImportError as imp_err:
#     ADD_SAFE_GLOBALS_MESSAGE = f"Warning: Could not import one or more necessary items for torch compatibility: {imp_err}"
# except Exception as e:
#     ADD_SAFE_GLOBALS_MESSAGE = f"Warning: An unexpected error occurred while adding safe globals for torch: {e}"
# --- End of allowlist section ---


def load_data():
    """Loads GA4 data from an uploaded CSV file."""
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("CSV must contain 'Date' and 'Sessions' columns.")
                return None
            if df.empty:
                 st.error("Uploaded CSV file is empty.")
                 return None
            try:
                import numpy
            except ImportError:
                st.error("NumPy library is required but not installed. Please install it (`pip install numpy`).")
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
    """
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    except ValueError:
        st.error("Error parsing 'Date' column. Ensure it's in YYYYMMDD format.")
        return None, None
    except KeyError:
         st.error("CSV file must contain a 'Date' column.")
         return None, None
    except Exception as e:
         st.error(f"An error occurred during date conversion: {e}")
         return None, None

    if 'Sessions' not in df.columns:
        st.error("CSV file must contain a 'Sessions' column.")
        return None, None

    df = df.sort_values('Date')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=['ds', 'y'], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        st.warning(f"Dropped {dropped_rows} rows with missing dates or non-numeric session values.")

    if df.empty:
        st.error("No valid data remaining after processing.")
        return None, None

    last_date = df['ds'].max()
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date in the data.")
        st.error(f"Last observed date: {last_date.date()}. Selected end date: {forecast_end_date.date()}")
        return None, last_date

    # --- NeuralProphet Section ---
    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        quantiles=[0.05, 0.95] # We know these were requested
    )

    st.info("Training NeuralProphet model... this may take a few moments.")
    progress_bar = st.progress(0)
    forecast = None

    try:
        min_data_points = 30
        if hasattr(model, 'n_lags') and model.n_lags > 0:
             min_data_points = max(min_data_points, model.n_lags * 2 + 14)

        if len(df) < min_data_points:
             st.warning(f"Dataset has only {len(df)} data points (recommended: {min_data_points}+). Results might be less reliable.")
             df_train = df
             metrics = model.fit(df_train, freq="D")
             progress_bar.progress(50)
        else:
            df_train, df_val = model.split_df(df, freq="D", valid_p=0.1)
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

        future = model.make_future_dataframe(df=df, periods=periods)
        progress_bar.progress(75)
        forecast = model.predict(future)
        progress_bar.progress(100)

    except Exception as e:
        st.error(f"Error during NeuralProphet fitting or prediction: {e}")
        import traceback
        traceback.print_exc()
        progress_bar.progress(100)
        return None, last_date
    # --- End NeuralProphet Section ---

    if forecast is None or forecast.empty or 'ds' not in forecast.columns or 'yhat1' not in forecast.columns:
         st.error("Prediction failed or returned unexpected results.")
         return None, last_date

    # --- Plotting Section ---
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['ds'], df['y'], label='Actual', color='blue', marker='.', markersize=4, linestyle='-')
    ax.plot(forecast['ds'], forecast['yhat1'], label='Forecast (yhat1)', color='green', linestyle='--')

    # Plot uncertainty intervals using quantile columns
    # --- MODIFICATION: Hardcode column names ---
    # Using older PyTorch might change how NP exposes attributes.
    # We hardcode based on the known quantiles=[0.05, 0.95] used in initialization.
    lower_q_col = 'yhat1 5.0%'
    upper_q_col = 'yhat1 95.0%'
    uncertainty_label = 'Uncertainty Interval (90%)' # Corresponds to 0.95 - 0.05
    # --- END MODIFICATION ---

    if lower_q_col in forecast.columns and upper_q_col in forecast.columns:
        ax.fill_between(forecast['ds'],
                        forecast[lower_q_col],
                        forecast[upper_q_col],
                        color='green', alpha=0.2,
                        label=uncertainty_label) # Use the fixed label
    else:
        st.warning(f"Could not find uncertainty columns: '{lower_q_col}', '{upper_q_col}'. Plotting without intervals.")

    # --- Google Update Shading ---
    google_updates = [
        ('20230315', '20230328', 'Mar 23 Core'), ('20230822', '20230907', 'Aug 23 Core'),
        ('20230914', '20230928', 'Sep 23 Helpful'), ('20231004', '20231019', 'Oct 23 Core+Spam'),
        ('20231102', '20231204', 'Nov 23 Core+Spam'), ('20240305', '20240419', 'Mar 24 Core'),
        ('20240506', '20240507', 'Site Rep Abuse'), ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'Jun 24 Core'), ('20240815', '20240903', 'Aug 24 Core'),
        ('20241111', '20241205', 'Nov 24 Core'),
        ('20241212', '20241218', 'Dec 24 Core'),
        ('20241219', '20241226', 'Dec 24 Spam'),
        ('20250313', '20250327', 'Mar 25 Core')
    ]
    # ...(rest of google update plotting code remains the same)...
    plot_bottom, plot_top = ax.get_ylim()
    text_y_pos = plot_top * 1.02
    for start_str, end_str, label in google_updates:
        try:
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            if start_date <= forecast['ds'].max() and end_date >= df['ds'].min():
                ax.axvspan(start_date, end_date, color='lightcoral', alpha=0.2)
                mid_date = start_date + (end_date - start_date) / 2
                if mid_date >= ax.get_xlim()[0] and mid_date <= ax.get_xlim()[1]:
                     ax.text(mid_date, text_y_pos, label, ha='center', va='bottom', fontsize=8, rotation=90, color='dimgray')
        except Exception as e:
            st.warning(f"Could not plot Google Update '{label}': {e}")
    ax.set_ylim(bottom=plot_bottom, top=text_y_pos * 1.05)

    ax.set_title('Daily Actual vs. Forecasted GA4 Sessions (NeuralProphet) with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

    return forecast, last_date


def display_dashboard(forecast, last_date, forecast_end_date):
    """Displays the forecast data table and summary metrics."""
    st.subheader("Forecast Data Table")

    # --- MODIFICATION: Force fixed column names ---
    # Based on known quantiles=[0.05, 0.95] used in initialization.
    lower_q_col = 'yhat1 5.0%'
    upper_q_col = 'yhat1 95.0%'
    # --- END MODIFICATION ---

    # Check if quantile columns exist, handle gracefully if not
    if lower_q_col not in forecast.columns:
        forecast[lower_q_col] = pd.NA
        st.warning(f"Column '{lower_q_col}' not found in forecast data.")
    if upper_q_col not in forecast.columns:
        forecast[upper_q_col] = pd.NA
        st.warning(f"Column '{upper_q_col}' not found in forecast data.")

    # Filter forecast for the future period
    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)].copy()

    # Select and rename columns for display
    display_cols = ['ds', 'yhat1', lower_q_col, upper_q_col]
    display_cols = [col for col in display_cols if col in forecast_filtered.columns]
    forecast_display = forecast_filtered[display_cols]

    # Rename after selection
    rename_map = {
        'yhat1': 'Forecast',
        # Use the fixed names to generate labels correctly
        lower_q_col: f'Lower Bound ({lower_q_col.split(" ")[-1]})' if lower_q_col in forecast_display else 'Lower Bound',
        upper_q_col: f'Upper Bound ({upper_q_col.split(" ")[-1]})' if upper_q_col in forecast_display else 'Upper Bound'
    }
    forecast_display = forecast_display.rename(columns=rename_map)

    # Format date column and numbers
    forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
    for col in forecast_display.columns:
        if col != 'ds' and pd.api.types.is_numeric_dtype(forecast_display[col]):
             forecast_display[col] = forecast_display[col].map('{:,.0f}'.format, na_action='ignore')

    st.dataframe(forecast_display, use_container_width=True)

    # --- Forecast Summary ---
    st.subheader("Forecast Summary")
    horizon = (forecast_end_date - last_date).days
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon} days")

    forecast_future = forecast[forecast['ds'] > last_date]
    if forecast_future.empty:
        st.write("No forecast data available for the selected future date range.")
        return

    closest_idx = (forecast_future['ds'] - forecast_end_date).abs().idxmin()
    forecast_value = forecast_future.loc[closest_idx]

    delta_val = pd.NA
    delta_str = "Range N/A"
    # Use the fixed column names here too
    if pd.notna(forecast_value.get(lower_q_col)) and pd.notna(forecast_value.get(upper_q_col)):
         delta_val = forecast_value[upper_q_col] - forecast_value[lower_q_col]
         if pd.notna(delta_val):
              delta_str = f"Range: {int(delta_val):,}"

    metric_value = int(forecast_value['yhat1']) if pd.notna(forecast_value['yhat1']) else "N/A"
    metric_value_display = f"{metric_value:,}" if isinstance(metric_value, int) else metric_value

    st.metric(label=f"Forecasted Traffic (at {forecast_end_date.date()})",
              value=metric_value_display,
              delta=delta_str)

    # --- Year-over-Year Calculation ---
    st.subheader("Year-over-Year Comparison (Forecast vs Forecast)")
    start_forecast = last_date + pd.Timedelta(days=1)
    end_forecast = forecast_end_date
    current_period = forecast[(forecast['ds'] >= start_forecast) & (forecast['ds'] <= end_forecast)]

    start_prev = start_forecast - pd.Timedelta(days=365)
    end_prev = end_forecast - pd.Timedelta(days=365)
    prev_period = forecast[(forecast['ds'] >= start_prev) & (forecast['ds'] <= end_prev)]

    if not current_period.empty and not prev_period.empty:
        if len(current_period) != len(prev_period):
            st.warning(f"YoY Comparison Warning: Periods have different lengths ({len(current_period)} vs {len(prev_period)} days). Comparison might be skewed.")

        current_sum = current_period['yhat1'].sum()
        prev_sum = prev_period['yhat1'].sum()

        change_label = "N/A"
        if pd.notna(current_sum) and pd.notna(prev_sum):
            if prev_sum != 0:
                yoy_change = ((current_sum - prev_sum) / prev_sum) * 100
                change_label = f"{yoy_change:.2f}%"
            elif current_sum > 0:
                 change_label = "inf%"
            else:
                 change_label = "0.00%"

        st.write(f"Total Forecasted ({start_forecast.date()} to {end_forecast.date()}): {current_sum:,.0f}")
        st.write(f"Total Forecasted ({start_prev.date()} to {end_prev.date()}): {prev_sum:,.0f}")
        st.write(f"Year-over-Year Change: {change_label}")
    else:
        st.warning("Not enough historical forecast data within the current run for Year-over-Year calculation.")
        st.write(f"(Requires forecast data covering {start_prev.date()} to {end_prev.date()})")


def main():
    """Main function to run the Streamlit app."""
    if ADD_SAFE_GLOBALS_MESSAGE:
        st.info(ADD_SAFE_GLOBALS_MESSAGE) # Display message about safe_globals status

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
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    min_forecast_date = (pd.Timestamp.today() + timedelta(days=1)).date()
    forecast_end_date_input = st.sidebar.date_input(
        "Select Forecast End Date", value=default_forecast_end, min_value=min_forecast_date,
        help="Choose the date up to which you want to forecast." )
    forecast_end_date = pd.to_datetime(forecast_end_date_input)
    st.sidebar.markdown("---")
    # Updated requirements list in sidebar info
    st.sidebar.info("Ensure `neuralprophet`, `torch<2.6`, `numpy`, `pandas`, `matplotlib`, and `streamlit` are installed.")

    # --- Main Area ---
    df_original = load_data()

    if df_original is not None:
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df_original.head(), use_container_width=True)

        forecast_df, last_date = plot_daily_forecast(df_original.copy(), forecast_end_date)

        if forecast_df is not None and last_date is not None:
            display_dashboard(forecast_df, last_date, forecast_end_date)
            try:
                download_cols = [col for col in forecast_df.columns if 'yhat' in col or col == 'ds' or '%' in col]
                csv_data = forecast_df[download_cols].to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8')
                st.download_button(
                    label="💾 Download Full Forecast CSV", data=csv_data,
                    file_name=f'neuralprophet_forecast_{forecast_end_date.date()}.csv', mime='text/csv',
                    help="Downloads the forecast values and uncertainty bounds." )
            except Exception as e:
                st.error(f"Failed to generate download file: {e}")
        else:
            st.error("Forecasting could not be completed due to previous errors.")

    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()
