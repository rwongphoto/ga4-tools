import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import math
from datetime import timedelta
import google.generativeai as genai
import traceback
import os

# --- Google Gemini Configuration ---
# (configure_gemini function remains the same, using os.getenv)
def configure_gemini():
    """Configures the Gemini API using the GOOGLE_API_KEY environment variable."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔴 Error: GOOGLE_API_KEY environment variable not found or empty. Please configure it in Posit Connect.")
        st.info("💡 For local testing, you might need to set this environment variable in your terminal.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during Gemini configuration: {e}")
        return False

# --- NEW Gemini Function for Historical Analysis ---
def get_gemini_historical_analysis(actual_df, google_updates):
    """
    Analyzes the historical traffic data using Google Gemini API, focusing on
    potential correlations with Google Algorithm Updates.

    Args:
        actual_df (pd.DataFrame): DataFrame with historical data ('ds', 'y' columns).
        google_updates (list): List of tuples containing Google update info.

    Returns:
        str: The analysis summary from Gemini, or an error message.
    """
    if not configure_gemini():
        return "Gemini API not configured. Analysis cannot proceed."

    if actual_df.empty:
        return "Historical data is empty. Cannot perform analysis."

    try:
        # --- Prepare data for the prompt ---
        # Basic summary of the historical data
        start_date = actual_df['ds'].min().strftime('%Y-%m-%d')
        end_date = actual_df['ds'].max().strftime('%Y-%m-%d')
        overall_avg = int(actual_df['y'].mean())
        start_val = int(actual_df.iloc[0]['y'])
        end_val = int(actual_df.iloc[-1]['y'])
        overall_trend = "Increased" if end_val > start_val else "Decreased" if end_val < start_val else "Remained Stable"

        # Convert some data points around updates to include? (Optional, keep it simple first)
        # For now, just send summary stats.

        historical_summary_str = f"""
        - Data Period: {start_date} to {end_date}
        - Starting Sessions: {start_val}
        - Ending Sessions: {end_val}
        - Average Sessions: {overall_avg}
        - Overall Trend During Period: {overall_trend}
        """

        # Format Google Updates for the prompt
        updates_str = "\n".join([
            f"- {label} ({pd.to_datetime(start, format='%Y%m%d').strftime('%Y-%m-%d')} to {pd.to_datetime(end, format='%Y%m%d').strftime('%Y-%m-%d')})"
            for start, end, label in google_updates
        ])

        # --- Construct the prompt ---
        prompt = f"""
        Analyze the provided historical SEO traffic (sessions) data in the context of known Google Algorithm Updates.

        Context:
        - The data represents daily/weekly/monthly website sessions from Google Analytics.
        - The goal is to identify potential correlations between traffic fluctuations and the timing of Google Algorithm Updates.

        Historical Data Summary:
        {historical_summary_str}

        Google Algorithm Updates During or Near Data Period (YYYY-MM-DD):
        {updates_str}

        Task:
        Provide a concise analysis (around 3-5 bullet points or a short paragraph) summarizing observations about the historical traffic *in relation to the Google updates*. Focus on:
        1. Identifying any noticeable drops or increases in traffic volume that coincide (occur during or shortly after) specific Google update periods listed.
        2. Commenting on periods of increased volatility around update rollouts.
        3. Mentioning updates that seem to have had little to no observable impact on this dataset.
        4. Providing an overall assessment of how sensitive this website's traffic appears to be to the listed Google updates based *only* on the provided data summary and update list.
        5. Frame the analysis for an SEO Manager trying to understand past performance drivers. Do NOT analyze the future forecast.
        """

        # --- Call the Gemini API ---
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        analysis = response.text.replace('•', '*')
        return analysis

    except genai.types.generation_types.BlockedPromptException:
         st.error("🔴 Gemini API Error: The prompt was blocked. This might be due to safety settings.")
         return "Analysis failed: Prompt was blocked by safety filters."
    except Exception as e:
        st.error(f"🔴 An error occurred while generating the AI analysis: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return f"Analysis failed due to an error: {e}"

# --- load_data function remains the same ---
def load_data():
    uploaded_file = st.file_uploader(
        "Choose a GA4 CSV file",
        type="csv",
        key="ga4_csv_uploader" # Keep the unique key
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("🔴 Error: CSV must contain 'Date' and 'Sessions' columns.")
                return None
            else:
                 try:
                     df['Date'] = df['Date'].astype(str)
                     pd.to_datetime(df['Date'], format='%Y%m%d')
                     return df
                 except ValueError:
                     st.error("🔴 Error: 'Date' column contains values not in YYYYMMDD format.")
                     return None
                 except Exception as date_err:
                     st.error(f"🔴 Error processing 'Date' column: {date_err}")
                     return None
        except Exception as e:
            st.error(f"🔴 Error loading or parsing CSV: {e}")
            return None
    else:
        return None

# --- MODIFIED Plotting Functions (accept google_updates) ---
def plot_daily_forecast(df, forecast_end_date, google_updates): # Added google_updates arg
    # Prepare data (convert date, rename columns)
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
    except Exception as e:
        st.error(f"Error processing data for daily plot: {e}")
        return None, None, None # Return Nones on error

    last_date = df['ds'].max()
    if df.empty or last_date is pd.NaT:
         st.error("Input data is empty or last date cannot be determined for daily plot.")
         return None, None, None
    last_row = df[df['ds'] == last_date]
    if last_row.empty:
        st.error(f"No daily data found for the last date {last_date}. Check data integrity.")
        return None, last_date, None
    last_actual_value = last_row['y'].iloc[0]

    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date for daily forecast.")
        return None, last_date, last_actual_value # Still return values for potential historical analysis

    # Prophet modeling and prediction
    try:
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq='D')
        forecast = model.predict(future)
    except Exception as e:
        st.error(f"Error during Prophet forecasting (daily): {e}")
        return None, last_date, last_actual_value

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 8))
    try:
        ax.plot(df['ds'], df['y'], label='Actual', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='green')

        # Use passed google_updates list
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2
                ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
                y_limits = ax.get_ylim()
                text_y_pos = y_limits[1] * 0.98 if y_limits and len(y_limits) == 2 and y_limits[1] > y_limits[0] else (forecast['yhat'].max() * 0.98 if not forecast.empty else 0)
                ax.text(mid_date, text_y_pos, label, ha='center', va='top', fontsize=8, rotation=90)
            except Exception as plot_err:
                 st.warning(f"Could not plot Google update '{label}': {plot_err}")

        ax.set_title('Daily Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sessions')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during plotting (daily): {e}")
    finally:
        plt.close(fig) # Ensure figure is closed

    return forecast, last_date, last_actual_value

def plot_weekly_forecast(df, forecast_end_date, google_updates): # Added google_updates arg
    # Prepare data
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
        df_indexed = df.set_index('ds')
        df_weekly = df_indexed.resample('W').sum().reset_index()
    except Exception as e:
        st.error(f"Error processing data for weekly plot: {e}")
        return None, None, None

    if df_weekly.empty or df_weekly['ds'].max() is pd.NaT:
         st.error("No weekly data or last date invalid after resampling.")
         return None, None, None
    last_date = df_weekly['ds'].max()
    last_actual_value = df_weekly[df_weekly['ds'] == last_date]['y'].iloc[0]

    periods = math.ceil((forecast_end_date - last_date).days / 7)
    if periods <= 0:
        st.error("Forecast end date must be after the last observed date for weekly forecast.")
        return None, last_date, last_actual_value

    # Prophet modeling
    try:
        model = Prophet()
        model.fit(df_weekly)
        future = model.make_future_dataframe(periods=periods, freq='W')
        forecast = model.predict(future)
    except Exception as e:
        st.error(f"Error during Prophet forecasting (weekly): {e}")
        return None, last_date, last_actual_value

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 8))
    try:
        ax.plot(df_weekly['ds'], df_weekly['y'], label='Weekly Actual', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='Weekly Forecast', color='green')

        # Use passed google_updates list
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2
                ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
                y_limits = ax.get_ylim()
                text_y_pos = y_limits[1] * 0.98 if y_limits and len(y_limits) == 2 and y_limits[1] > y_limits[0] else (forecast['yhat'].max() * 0.98 if not forecast.empty else 0)
                ax.text(mid_date, text_y_pos, label, ha='center', va='top', fontsize=8, rotation=90)
            except Exception as plot_err:
                 st.warning(f"Could not plot Google update '{label}': {plot_err}")

        ax.set_title('Weekly Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sessions (Weekly)')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during plotting (weekly): {e}")
    finally:
        plt.close(fig)

    return forecast, last_date, last_actual_value

def plot_monthly_forecast(df, forecast_end_date, google_updates): # Added google_updates arg
    # Prepare data
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
        df_indexed = df.set_index('ds')
        df_monthly = df_indexed.resample('M').sum().reset_index()
    except Exception as e:
        st.error(f"Error processing data for monthly plot: {e}")
        return None, None, None

    if df_monthly.empty or df_monthly['ds'].max() is pd.NaT:
         st.error("No monthly data or last date invalid after resampling.")
         return None, None, None
    last_date = df_monthly['ds'].max()
    last_actual_value = df_monthly[df_monthly['ds'] == last_date]['y'].iloc[0]

    periods = 0
    temp_date = last_date
    while temp_date < forecast_end_date:
        temp_date += pd.offsets.MonthEnd(1)
        periods += 1

    if periods <= 0 and forecast_end_date <= last_date:
        st.error("Forecast end date must be after the last observed month-end date for monthly forecast.")
        return None, last_date, last_actual_value

    # Prophet modeling
    try:
        model = Prophet()
        model.fit(df_monthly)
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
    except Exception as e:
        st.error(f"Error during Prophet forecasting (monthly): {e}")
        return None, last_date, last_actual_value

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 8))
    try:
        ax.plot(df_monthly['ds'], df_monthly['y'], label='Monthly Actual', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='Monthly Forecast', color='green')

        # Use passed google_updates list
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2
                ax.axvspan(start_date, end_date, color='gray', alpha=0.2)
                y_limits = ax.get_ylim()
                text_y_pos = y_limits[1] * 0.98 if y_limits and len(y_limits) == 2 and y_limits[1] > y_limits[0] else (forecast['yhat'].max() * 0.98 if not forecast.empty else 0)
                ax.text(mid_date, text_y_pos, label, ha='center', va='top', fontsize=8, rotation=90)
            except Exception as plot_err:
                 st.warning(f"Could not plot Google update '{label}': {plot_err}")

        ax.set_title('Monthly Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sessions (Monthly)')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during plotting (monthly): {e}")
    finally:
        plt.close(fig)

    return forecast, last_date, last_actual_value

# --- display_dashboard remains the same (analyzes forecast output) ---
def display_dashboard(forecast, last_date, forecast_end_date, forecast_type):
    st.subheader("Forecast Data Table & Summary") # Combine header
    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]

    if not forecast_filtered.empty:
        st.dataframe(forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].astype({'yhat':'int', 'yhat_lower':'int', 'yhat_upper':'int'}))
    else:
        st.write("No forecast data in the selected future range.")

    horizon_str = ""
    if not forecast_filtered.empty:
        horizon = len(forecast_filtered)
        if forecast_type == "Daily Forecast": horizon_str = f"{horizon} days"
        elif forecast_type == "Weekly Forecast": horizon_str = f"{horizon} weeks"
        else: horizon_str = f"{horizon} months"
    else:
        horizon_str = "N/A"

    # Display summary metrics only if forecast is available
    if not forecast_filtered.empty:
        forecast_value_at_end = forecast_filtered.iloc[-1]
        forecast_range = int(forecast_value_at_end['yhat_upper'] - forecast_value_at_end['yhat_lower'])
        delta_val = forecast_range / 2

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Last Actual Date", value=f"{last_date.date()}")
        with col2:
            st.metric(label="Forecast Horizon", value=f"{horizon_str}")
        with col3:
            st.metric(label=f"Forecast at {forecast_value_at_end['ds'].date()}",
                      value=int(forecast_value_at_end['yhat']),
                      delta=f"±{delta_val:.0f} (Range: {forecast_range})",
                      delta_color="off")
    else:
        st.metric(label="Last Actual Date", value=f"{last_date.date() if last_date else 'N/A'}")
        st.metric(label="Forecast Horizon", value=horizon_str)
        st.metric(label="Forecast Value", value="N/A")


# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Google Algorithm Impact Visualizer + AI Analysis", layout="wide")
    st.title("📈 Google Algorithm Impact Visualizer with AI Analysis")
    st.write("""
        Upload GA4 sessions CSV ('Date' as YYYYMMDD, 'Sessions'). Visualize historical trends against Google updates,
        generate a Prophet forecast, and get AI analysis of past performance related to updates.
    """)
    st.info("💡 Ensure your CSV has 'Date' (format YYYYMMDD) and 'Sessions' columns.")

    # --- Define Google Updates List Early ---
    google_updates = [
        ('20230315', '20230328', 'Mar 2023 Core Update'), ('20230822', '20230907', 'Aug 2023 Core Update'),
        ('20230914', '20230928', 'Sept 2023 Helpful Content Update'), ('20231004', '20231019', 'Oct 2023 Core & Spam Updates'),
        ('20231102', '20231204', 'Nov 2023 Core & Spam Updates'), ('20240305', '20240419', 'Mar 2024 Core Update'),
        ('20240506', '20240507', 'Site Rep Abuse'), ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'June 2024 Core Update'), ('20240815', '20240903', 'Aug 2024 Core Update'),
        ('20241111', '20241205', 'Nov 2024 Core Update'), ('20241212', '20241218', 'Dec 2024 Core Update'),
        ('20241219', '20241226', 'Dec 2024 Spam Update'), ('20250313', '20250327', 'Mar 2025 Core Update')
    ]

    # --- Sidebar Controls ---
    forecast_type = st.sidebar.radio("Select Forecast Granularity", ("Daily", "Weekly", "Monthly"), key="forecast_type_radio") # Simplified labels
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end, min_value=pd.Timestamp.today().date(), key="forecast_date_input")
    forecast_end_date = pd.to_datetime(forecast_end_date_input)

    # --- File Upload and Initial Data Processing ---
    df_original = load_data()
    processed_df = None # To store df with 'ds' and 'y' columns

    if df_original is not None:
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df_original.head())

        # Prepare df for analysis and plotting (do this once)
        processed_df = df_original.copy()
        try:
            processed_df['Date'] = pd.to_datetime(processed_df['Date'].astype(str), format='%Y%m%d')
            processed_df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
            processed_df = processed_df.sort_values('ds').reset_index(drop=True) # Ensure sorted
        except Exception as e:
            st.error(f"Failed to process loaded data for analysis/plotting: {e}")
            processed_df = None # Invalidate if error

    # --- Main Area ---
    if processed_df is not None: # Only proceed if data is loaded and processed

        # --- AI Historical Analysis Section ---
        st.markdown("---")
        st.header("🤖 AI Historical Analysis vs. Google Updates")
        api_key_present = bool(os.getenv("GOOGLE_API_KEY"))
        if not api_key_present:
            st.warning("⚠️ GOOGLE_API_KEY environment variable not set. AI analysis disabled.")
        else:
            if st.button("📊 Analyze Historical Traffic vs. Updates", key="analyze_historical_button"):
                if configure_gemini():
                    with st.spinner("🧠 Analyzing historical data with Gemini..."):
                        # Pass the correctly processed DataFrame
                        historical_analysis_result = get_gemini_historical_analysis(processed_df, google_updates)
                    st.markdown(historical_analysis_result)
                # else: configure_gemini showed error

        # --- Forecasting and Plotting Section ---
        st.markdown("---")
        st.header("📊 Forecast & Visualization")

        forecast = None
        last_date = None
        last_actual_value = None

        # Determine plotting function based on selected granularity
        plot_function = None
        df_plot_copy = processed_df.copy() # Use the already processed df, make a copy
        granularity_label = forecast_type # e.g., "Daily"

        try:
            if granularity_label == "Daily":
                 plot_function = plot_daily_forecast
                 # Data is already daily
            elif granularity_label == "Weekly":
                 plot_function = plot_weekly_forecast
                 # Need to pass the original-like structure before resampling
                 df_plot_copy = df_original.copy() # Start from original for weekly/monthly plots
            elif granularity_label == "Monthly":
                 plot_function = plot_monthly_forecast
                 # Need to pass the original-like structure before resampling
                 df_plot_copy = df_original.copy() # Start from original for weekly/monthly plots

            # Generate Forecast and Plot
            with st.spinner(f"Generating {granularity_label} forecast and plot..."):
                 # Pass google_updates list to the plot function
                forecast_result = plot_function(df_plot_copy, forecast_end_date, google_updates)

                # Unpack results carefully, handle potential None returns
                if forecast_result:
                     forecast, last_date, last_actual_value = forecast_result
                else:
                     forecast, last_date, last_actual_value = None, None, None


            # Display Forecast Dashboard (only if forecast was successful)
            if forecast is not None and last_date is not None:
                st.markdown("---") # Separator before dashboard
                display_dashboard(forecast, last_date, forecast_end_date, f"{granularity_label} Forecast")
            elif last_date is not None: # If forecast failed but we have last_date
                 st.info("Forecasting failed, unable to display forecast summary.")
                 # Optionally display basic info like last date?
            # else: Plotting function itself likely showed an error


        except Exception as e:
            st.error(f"🔴 An error occurred during forecasting/plotting: {e}")
            st.error(f"Traceback: {traceback.format_exc()}")

    else:
        # Show message if no file uploaded or initial processing failed
        if 'ga4_csv_uploader' not in st.session_state or st.session_state.ga4_csv_uploader is None:
              st.info("Awaiting CSV file upload...")


    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()
