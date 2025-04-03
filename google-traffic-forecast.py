import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings

# Suppress specific warnings if they become noisy
# warnings.filterwarnings("ignore", message="The behavior of DatetimeProperties.to_pydatetime is deprecated")
# warnings.filterwarnings("ignore", category=FutureWarning)

# Model specific imports
from prophet import Prophet
from neuralprophet import NeuralProphet as NP
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA  # Example: Using AutoARIMA

# --- Configuration ---
# Define Google Update Ranges (centralized)
GOOGLE_UPDATES = [
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

# --- Data Loading and Preparation ---
def load_data():
    """Loads data from user-uploaded CSV file."""
    uploaded_file = st.file_uploader("Choose a GA4 CSV file (must contain 'Date' [YYYYMMDD] and 'Sessions' columns)", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Basic validation
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("CSV must contain 'Date' and 'Sessions' columns.")
                return None, None
            # Convert 'Date' column from string format (YYYYMMDD) to datetime
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df.sort_values('Date', inplace=True) # Ensure data is sorted
            last_date = df['Date'].max()
            return df, last_date
        except Exception as e:
            st.error(f"Error loading or processing file: {e}")
            return None, None
    else:
        st.info("Awaiting CSV file upload...")
        return None, None

# --- Forecasting Models ---
def run_prophet(df_hist, periods):
    """Runs Prophet forecast."""
    df_prophet = df_hist.rename(columns={'Date': 'ds', 'Sessions': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    # Prophet output already matches 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
    return forecast

def run_neural_prophet(df_hist, periods):
    """Runs NeuralProphet forecast."""
    df_np = df_hist.rename(columns={'Date': 'ds', 'Sessions': 'y'})
    # Initialize with quantiles for uncertainty intervals (e.g., 95% interval)
    model = NP(quantiles=[0.025, 0.975])
    # Fit the model
    metrics = model.fit(df_np, freq='D')
    # Create future dataframe
    future = model.make_future_dataframe(df_np, periods=periods)
    # Predict
    forecast = model.predict(future)
    # Standardize column names
    forecast.rename(columns={'yhat1': 'yhat',
                             'yhat1 2.5%': 'yhat_lower',
                             'yhat1 97.5%': 'yhat_upper'}, inplace=True)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def run_statsforecast(df_hist, periods):
    """Runs Statsforecast (AutoARIMA) forecast."""
    df_sf = df_hist.rename(columns={'Date': 'ds', 'Sessions': 'y'})
    df_sf['unique_id'] = 'GA4_Sessions' # Statsforecast needs a unique_id column

    # Define models (can add more like AutoETS(), AutoTheta())
    model_list = [AutoARIMA()]
    sf = StatsForecast(
        models=model_list,
        freq='D',  # Daily frequency
        # n_jobs=-1 # Use all available cores (optional)
    )

    # Fit and predict future periods (forecast method)
    # level=[95] corresponds to 0.025 and 0.975 quantiles
    forecast = sf.forecast(df=df_sf[['unique_id', 'ds', 'y']], h=periods, level=[95])

    # Standardize column names (adjust 'AutoARIMA' if using different/multiple models)
    model_name = 'AutoARIMA' # Change if using a different model
    forecast.rename(columns={model_name: 'yhat',
                             f'{model_name}-lo-95': 'yhat_lower',
                             f'{model_name}-hi-95': 'yhat_upper'}, inplace=True)

    # Select and return standardized columns
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# --- Plotting ---
def plot_forecast(df_actual, forecast, model_name, forecast_start_date):
    """Plots actual data vs forecast with Google update ranges."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot actual historical data
    ax.plot(df_actual['Date'], df_actual['Sessions'], label='Actual', color='blue', marker='.', linestyle='-')

    # Plot forecast (might start from the beginning or only future)
    ax.plot(forecast['ds'], forecast['yhat'], label=f'{model_name} Forecast', color='green', linestyle='--')

    # Plot confidence intervals if available
    if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
        # Only plot CI for the future forecast period for clarity
        future_forecast = forecast[forecast['ds'] >= forecast_start_date]
        ax.fill_between(future_forecast['ds'],
                        future_forecast['yhat_lower'],
                        future_forecast['yhat_upper'],
                        color='green', alpha=0.2, label='95% Confidence Interval')

    # Shade Google algorithm update ranges
    for start_str, end_str, label in GOOGLE_UPDATES:
        start_date = pd.to_datetime(start_str, format='%Y%m%d')
        end_date = pd.to_datetime(end_str, format='%Y%m%d')
        ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
        mid_date = start_date + (end_date - start_date) / 2
        # Adjust text position based on y-axis limits dynamically
        y_lim = ax.get_ylim()
        text_y_pos = y_lim[1] * 1.01 # Place slightly above the max y-limit
        ax.text(mid_date, text_y_pos, label, ha='center', va='bottom', fontsize=9, rotation=0) # Rotate if needed

    ax.set_title(f'Daily Actual vs. Forecasted GA4 Sessions ({model_name}) with Google Update Ranges')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sessions')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# --- Dashboard Display ---
def display_dashboard(forecast, df_actual, last_date, forecast_end_date):
    """Displays the forecast data table, summary metrics, and YoY comparison."""
    st.subheader("Forecast Data Table (Future Period)")

    # Filter forecast for FUTURE dates only for the table display
    forecast_future = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)].copy()

    if forecast_future.empty:
        st.warning("No future forecast data available in the selected range to display in the table.")
    else:
        # Round numeric columns for display
        display_cols = ['ds', 'yhat']
        if 'yhat_lower' in forecast_future.columns: display_cols.append('yhat_lower')
        if 'yhat_upper' in forecast_future.columns: display_cols.append('yhat_upper')
        st.dataframe(forecast_future[display_cols].round(2)) # Use round() for better readability

    # Calculate forecast horizon
    horizon = (forecast_end_date - last_date).days
    st.subheader("Forecast Summary")
    st.write(f"Forecast End Date: {forecast_end_date.date()}")
    st.write(f"Forecast Horizon: {horizon} days")

    # Get the forecast row closest to the forecast end date from the FUTURE predictions
    if not forecast_future.empty:
        closest_forecast_row = forecast_future.iloc[-1] # Get the last row of the future forecast
        metric_value = int(closest_forecast_row['yhat'])
        # Check if confidence intervals exist before creating delta string
        if 'yhat_lower' in closest_forecast_row and 'yhat_upper' in closest_forecast_row and \
           pd.notna(closest_forecast_row['yhat_lower']) and pd.notna(closest_forecast_row['yhat_upper']):
            delta_value = f"95% Range: {int(closest_forecast_row['yhat_lower'])} - {int(closest_forecast_row['yhat_upper'])}"
        else:
            delta_value = "CI not available"
        st.metric(label="Forecasted Sessions (at End Date)", value=metric_value, delta=delta_value)
    else:
        st.write("No forecast data available to show summary metric.")


    # --- Year-over-Year Calculation (Revised) ---
    st.subheader("Year-over-Year Comparison")
    start_forecast_period = last_date + pd.Timedelta(days=1)
    end_forecast_period = forecast_end_date

    # Sum forecasted traffic in the selected future period
    current_period_forecast_data = forecast_future[(forecast_future['ds'] >= start_forecast_period) & (forecast_future['ds'] <= end_forecast_period)]

    if not current_period_forecast_data.empty:
        current_sum_forecast = current_period_forecast_data['yhat'].sum()

        # Define the corresponding period one year earlier using ACTUAL data
        start_prev_actual = start_forecast_period - pd.Timedelta(days=365)
        end_prev_actual = end_forecast_period - pd.Timedelta(days=365)

        # Use df_actual (original data) for the previous period
        prev_period_actual_data = df_actual[(df_actual['Date'] >= start_prev_actual) & (df_actual['Date'] <= end_prev_actual)]

        if not prev_period_actual_data.empty:
            prev_sum_actual = prev_period_actual_data['Sessions'].sum()
            if prev_sum_actual != 0:
                yoy_change = ((current_sum_forecast - prev_sum_actual) / prev_sum_actual) * 100
            else:
                yoy_change = float('inf') # Avoid division by zero

            st.write(f"Total **Forecasted** Sessions ({start_forecast_period.date()} to {end_forecast_period.date()}): {current_sum_forecast:,.0f}")
            st.write(f"Total **Actual** Sessions ({start_prev_actual.date()} to {end_prev_actual.date()}): {prev_sum_actual:,.0f}")
            st.write(f"Year-over-Year Change: **{yoy_change:.2f}%**")
        else:
            st.write(f"Total **Forecasted** Sessions ({start_forecast_period.date()} to {end_forecast_period.date()}): {current_sum_forecast:,.0f}")
            st.warning(f"Not enough historical data ({start_prev_actual.date()} to {end_prev_actual.date()}) for Year-over-Year comparison.")
    else:
        st.warning("No forecast data available in the selected future period for Year-over-Year calculation.")

# --- Main App Logic ---
def main():
    st.set_page_config(layout="wide") # Use wider layout
    st.title("GA4 Daily Forecasting Tool")
    st.write("""
        Upload your Google Analytics 4 daily sessions data (CSV format) to generate forecasts using different models.
        The CSV file must have columns named **'Date'** (in YYYYMMDD format, e.g., 20231026) and **'Sessions'**.
        Ensure the data is sorted chronologically with the oldest date first.
    """)

    # --- Sidebar Controls ---
    st.sidebar.header("Forecasting Settings")

    # Model Selection
    model_options = ['Prophet', 'NeuralProphet', 'Statsforecast (AutoARIMA)']
    selected_model = st.sidebar.selectbox("Choose Forecasting Model", model_options)

    # Forecast End Date
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end, min_value=pd.Timestamp.today().date())
    forecast_end_date = pd.to_datetime(forecast_end_date_input)

    # --- Main Panel ---
    df_raw, last_date = load_data()

    if df_raw is not None and last_date is not None:
        st.subheader("Data Preview (Last 5 Rows)")
        st.dataframe(df_raw.tail())

        # Calculate forecast periods
        periods = (forecast_end_date - last_date).days
        if periods <= 0:
            st.error("Forecast end date must be after the last observed date in the data.")
            st.stop() # Stop execution if date is invalid

        try:
            # Run selected model
            st.info(f"Generating forecast using {selected_model} for {periods} days...")
            forecast_df = None
            with st.spinner(f"Running {selected_model}... this may take a moment."):
                if selected_model == 'Prophet':
                    forecast_df = run_prophet(df_raw.copy(), periods) # Use copy to be safe
                elif selected_model == 'NeuralProphet':
                    forecast_df = run_neural_prophet(df_raw.copy(), periods)
                elif selected_model == 'Statsforecast (AutoARIMA)':
                    forecast_df = run_statsforecast(df_raw.copy(), periods)
                    # Note: Statsforecast `forecast` method returns only future dates.
                    # We need to prepend historical actuals OR handle plotting appropriately.
                    # For simplicity in this version, plot_forecast handles plotting actuals separately.
                    # The returned forecast_df here contains ONLY future predictions.

            st.success(f"{selected_model} forecast complete!")

            # --- Display Results ---
            forecast_start_date = last_date + timedelta(days=1)
            plot_forecast(df_raw, forecast_df, selected_model, forecast_start_date)
            display_dashboard(forecast_df, df_raw, last_date, forecast_end_date) # Pass df_raw for YoY actuals

            # --- Download Button ---
            # Offer download of the generated forecast data (might be only future for some models)
            csv_data = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download {selected_model} Forecast CSV",
                data=csv_data,
                file_name=f'{selected_model.lower().replace(" ", "_")}_forecast.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"An error occurred during forecasting with {selected_model}:")
            st.exception(e) # Display the full error traceback for debugging

    # Footer link
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()

