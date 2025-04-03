import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
from datetime import timedelta
import logging
import torch # Import torch
from torch import serialization # Import serialization module
# Import the specific class mentioned in the error
from neuralprophet.configure import ConfigSeasonality, ConfigLags # Added ConfigLags as it might be needed too

# Optional: Suppress excessive logging
set_log_level("ERROR")

# --- Add this section ---
# Allowlist necessary NeuralProphet classes for torch.load with weights_only=True (PyTorch 2.6+)
# This addresses the "Weights only load failed" error.
try:
    # Add classes that might be pickled/unpickled by NeuralProphet internally
    # Start with the one from the error, add others if new errors appear.
    safe_globals_list = [ConfigSeasonality]
    # It's possible other config classes might be needed depending on model config/internal state
    # Example: from neuralprophet.configure import ConfigTrend, ConfigLags, ConfigAr
    # safe_globals_list.extend([ConfigTrend, ConfigLags, ConfigAr]) # Uncomment/add as needed

    serialization.add_safe_globals(safe_globals_list)
    print(f"Successfully added {len(safe_globals_list)} NeuralProphet class(es) to torch safe globals.")
except AttributeError:
    # Handle cases where serialization.add_safe_globals might not exist (e.g., older PyTorch)
    print("Note: torch.serialization.add_safe_globals not found or not needed for this PyTorch version.")
except ImportError:
    print("Warning: Could not import necessary NeuralProphet configure classes.")
except Exception as e:
    # Catch any other unexpected errors during the process
    st.warning(f"An unexpected error occurred while adding safe globals for torch: {e}")
# --- End of added section ---


def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
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
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    except ValueError:
        st.error("Error parsing 'Date' column. Ensure it's in YYYYMMDD format.")
        return None, None
    except Exception as e:
         st.error(f"An error occurred during date conversion: {e}")
         return None, None

    df = df.sort_values('Date')
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    df.dropna(subset=['ds', 'y'], inplace=True)
    if df.empty:
        st.error("No valid data remaining after processing.")
        return None, None

    last_date = df['ds'].max()
    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date.")
        return None, last_date

    # Initialize NeuralProphet model
    # Adding n_lags based on typical daily data patterns (e.g., look back 7 days)
    # You might need to adjust n_lags based on your specific data patterns
    # If n_lags > 0, you might need ConfigLags in the safe_globals list above
    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        # n_lags=7, # Example: Add autoregression - uncomment if desired
        quantiles=[0.05, 0.95]
    )
    # If using n_lags > 0, ensure ConfigLags is imported and added to safe_globals
    # from neuralprophet.configure import ConfigLags
    # serialization.add_safe_globals([ConfigSeasonality, ConfigLags]) # Example

    st.info("Training NeuralProphet model...")
    progress_bar = st.progress(0)
    try:
        # Ensure df has enough data points if using lags or validation split
        min_data_points = 14 # Arbitrary minimum, adjust as needed
        if model.n_lags:
             min_data_points = max(min_data_points, model.n_lags + 7) # Need enough data for lags + buffer

        if len(df) < min_data_points:
             st.warning(f"Dataset has only {len(df)} points. Consider using more data for better results, especially if using lags.")
             # Proceeding anyway, but model quality might be affected
             df_train = df # Use all data for training if too small to split
             metrics = model.fit(df_train, freq="D")

        else:
            df_train, df_val = model.split_df(df, freq="D", valid_p=0.1)
            metrics = model.fit(df_train, freq="D") # Fit on training data
            # Optional: Evaluate on validation set
            # val_metrics = model.test(df_val)
            # st.write("Model Validation Metrics:")
            # st.write(val_metrics)


        progress_bar.progress(50) # Mid-point after fitting
        st.write("Model Training Metrics:")
        st.write(metrics)

        # Create future dataframe & predict
        future = model.make_future_dataframe(df=df, periods=periods)
        progress_bar.progress(75)
        forecast = model.predict(future)
        progress_bar.progress(100)

    except Exception as e:
        st.error(f"Error during NeuralProphet fitting or prediction: {e}")
        # Print detailed traceback to console for debugging if running locally
        import traceback
        traceback.print_exc()
        progress_bar.progress(100)
        return None, last_date

    if forecast is None or forecast.empty or 'ds' not in forecast.columns or 'yhat1' not in forecast.columns:
         st.error("Prediction failed or returned unexpected results.")
         return None, last_date

    # --- Plotting Section (Adjusted for NeuralProphet output) ---
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df['ds'], df['y'], label='Actual', color='blue', marker='.', linestyle='-')

    ax.plot(forecast['ds'], forecast['yhat1'], label='Forecast (yhat1)', color='green')

    lower_q_col = f'yhat1 {model.quantiles[0]*100:.1f}%'
    upper_q_col = f'yhat1 {model.quantiles[1]*100:.1f}%'

    if lower_q_col in forecast.columns and upper_q_col in forecast.columns:
        ax.fill_between(forecast['ds'],
                        forecast[lower_q_col],
                        forecast[upper_q_col],
                        color='green', alpha=0.2, label='Uncertainty Interval (90%)')
    else:
        st.warning(f"Could not find uncertainty columns: '{lower_q_col}', '{upper_q_col}'. Plotting without intervals.")

    # --- Google Update Shading ---
    google_updates = [
        ('20230315', '20230328', 'Mar 2023 Core Update'), ('20230822', '20230907', 'Aug 2023 Core Update'),
        ('20230914', '20230928', 'Sept 2023 Helpful Content'), ('20231004', '20231019', 'Oct 2023 Core & Spam'),
        ('20231102', '20231204', 'Nov 2023 Core & Spam'), ('20240305', '20240419', 'Mar 2024 Core Update'),
        ('20240506', '20240507', 'Site Rep Abuse'), ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'June 2024 Core'), ('20240815', '20240903', 'Aug 2024 Core'),
        ('20241111', '20241205', 'Nov 2024 Core'), ('20241212', '20241218', 'Dec 2024 Core'),
        ('20241219', '20241226', 'Dec 2024 Spam'), ('20250313', '20250327', 'Mar 2025 Core')
    ]
    plot_bottom, plot_top = ax.get_ylim()
    for start_str, end_str, label in google_updates:
        try:
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            ax.axvspan(start_date, end_date, color='gray', alpha=0.15) # Slightly lighter alpha
            mid_date = start_date + (end_date - start_date) / 2
            # Rotate text and place slightly above data max
            text_y_pos = plot_top * 1.01 # Place text just above the max y-limit
            ax.text(mid_date, text_y_pos, label, ha='center', va='bottom', fontsize=8, rotation=90)
        except Exception as e:
            st.warning(f"Could not plot Google Update '{label}': {e}")

    # Adjust y-limits after adding text to ensure visibility
    final_bottom, final_top = ax.get_ylim()
    ax.set_ylim(final_bottom, final_top * 1.1) # Add 10% padding at the top for text

    ax.set_title('Daily Actual vs. Forecasted GA4 Sessions (NeuralProphet) with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend(loc='upper left') # Adjust legend position if needed
    plt.tight_layout()
    st.pyplot(fig)

    return forecast, last_date


def display_dashboard(forecast, last_date, forecast_end_date):
    st.subheader("Forecast Data Table")

    # Dynamically get quantile column names based on model's actual quantiles
    if hasattr(forecast, 'columns'):
        q_cols = [col for col in forecast.columns if '%' in col and 'yhat1' in col]
        lower_q_col = q_cols[0] if len(q_cols) > 0 else 'yhat1 5.0%' # Fallback default
        upper_q_col = q_cols[-1] if len(q_cols) > 0 else 'yhat1 95.0%' # Fallback default
    else: # Fallback if forecast object isn't as expected
        lower_q_col = 'yhat1 5.0%'
        upper_q_col = 'yhat1 95.0%'

    # Check if quantile columns exist
    if lower_q_col not in forecast.columns:
        forecast[lower_q_col] = None
        st.warning(f"Column '{lower_q_col}' not found in forecast data.")
    if upper_q_col not in forecast.columns:
        forecast[upper_q_col] = None
        st.warning(f"Column '{upper_q_col}' not found in forecast data.")

    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]
    display_cols = ['ds', 'yhat1', lower_q_col, upper_q_col]
    forecast_display = forecast_filtered[display_cols].rename(columns={
        'yhat1': 'Forecast',
        lower_q_col: f'Lower Bound ({lower_q_col.split(" ")[-1]})', # Dynamic label
        upper_q_col: f'Upper Bound ({upper_q_col.split(" ")[-1]})'  # Dynamic label
    })
    # Format date column for better readability
    forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
    st.dataframe(forecast_display)

    horizon = (forecast_end_date - last_date).days
    st.subheader("Forecast Summary")
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon} days")

    forecast_future = forecast[forecast['ds'] > last_date]
    if forecast_future.empty:
        st.write("No forecast data available for the selected date range.")
        return

    closest_idx = (forecast_future['ds'] - forecast_end_date).abs().idxmin()
    forecast_value = forecast_future.loc[closest_idx]

    delta_val = None
    delta_str = "Range N/A"
    if pd.notna(forecast_value[lower_q_col]) and pd.notna(forecast_value[upper_q_col]):
         delta_val = forecast_value[upper_q_col] - forecast_value[lower_q_col]
         # Check if delta_val is not NaN before formatting
         if pd.notna(delta_val):
              delta_str = f"Range: {int(delta_val)}"


    # Check if forecast value is not NaN before formatting
    metric_value = int(forecast_value['yhat1']) if pd.notna(forecast_value['yhat1']) else "N/A"

    st.metric(label="Forecasted Traffic (at End Date)", value=metric_value, delta=delta_str)

    # Year-over-Year Calculation
    start_forecast = last_date + pd.Timedelta(days=1)
    end_forecast = forecast_end_date
    current_period = forecast[(forecast['ds'] >= start_forecast) & (forecast['ds'] <= end_forecast)]

    start_prev = start_forecast - pd.Timedelta(days=365)
    end_prev = end_forecast - pd.Timedelta(days=365)
    prev_period = forecast[(forecast['ds'] >= start_prev) & (forecast['ds'] <= end_prev)]

    if not current_period.empty and not prev_period.empty:
        current_sum = current_period['yhat1'].sum()
        prev_sum = prev_period['yhat1'].sum()
        if pd.notna(prev_sum) and prev_sum != 0:
            yoy_change = ((current_sum - prev_sum) / prev_sum) * 100
            change_label = f"{yoy_change:.2f}%"
        elif pd.notna(prev_sum) and prev_sum == 0 and current_sum > 0:
             change_label = "inf%"
        elif pd.notna(prev_sum) and prev_sum == 0 and current_sum == 0:
             change_label = "0.00%"
        else: # Handle cases where prev_sum might be NaN
             change_label = "N/A"


        st.subheader("Year-over-Year Comparison (Forecast vs Forecast)")
        st.write(f"Total Forecasted Traffic ({start_forecast.date()} to {end_forecast.date()}): {current_sum:.0f}")
        st.write(f"Total Forecasted Traffic ({start_prev.date()} to {end_prev.date()}): {prev_sum:.0f}")
        st.write(f"Year-over-Year Change: {change_label}")
    else:
        st.warning("Not enough historical forecast data within the current run for Year-over-Year calculation.")
        st.write(f"Required historical forecast range: {start_prev.date()} to {end_prev.date()}")


def main():
    st.set_page_config(layout="wide")
    st.title("GA4 Daily Forecasting with NeuralProphet")
    st.write("""
        This app loads GA4 data (CSV), fits a **NeuralProphet** model to forecast daily sessions,
        and displays actual vs. forecasted traffic with shaded Google update ranges.
        A summary dashboard with a year-over-year comparison is provided below.

        **Requirements:**
        - CSV file must have columns named "Date" (format YYYYMMDD) and "Sessions".
        - Data should be sorted chronologically (oldest first is best).
        - Ensure you have the `neuralprophet`, `torch`, and `pandas` libraries installed (`pip install neuralprophet torch pandas matplotlib streamlit`).
    """)

    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end, min_value=pd.Timestamp.today().date())
    forecast_end_date = pd.to_datetime(forecast_end_date_input)

    df = load_data()
    if df is not None:
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df.head())

        if len(df) < 14:
             st.warning(f"Warning: Short dataset ({len(df)} rows). Model performance might be limited.")

        forecast_df, last_date = plot_daily_forecast(df.copy(), forecast_end_date)

        if forecast_df is not None and last_date is not None:
            display_dashboard(forecast_df, last_date, forecast_end_date)

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

    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()
