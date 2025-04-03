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
def configure_gemini():
    """Configures the Gemini API using the GOOGLE_API_KEY environment variable."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🔴 Error: GOOGLE_API_KEY environment variable not found or empty. Please configure it in Posit Connect.")
        # st.info("💡 For local testing, you might need to set this environment variable in your terminal.") # Keep UI cleaner
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during Gemini configuration: {e}")
        return False

# --- Gemini Function for Historical Analysis ---
def get_gemini_historical_analysis(actual_df_processed, google_updates): # Expects df with 'ds', 'y'
    """
    Analyzes the historical traffic data using Google Gemini API, focusing on
    potential correlations with Google Algorithm Updates.

    Args:
        actual_df_processed (pd.DataFrame): DataFrame with historical data ('ds', 'y' columns, sorted by ds).
        google_updates (list): List of tuples containing Google update info.

    Returns:
        str: The analysis summary from Gemini, or an error message.
    """
    if not configure_gemini():
        return "Gemini API not configured. Analysis cannot proceed."

    if actual_df_processed.empty:
        return "Historical data is empty. Cannot perform analysis."

    try:
        # --- Prepare data for the prompt ---
        start_date = actual_df_processed['ds'].min().strftime('%Y-%m-%d')
        end_date = actual_df_processed['ds'].max().strftime('%Y-%m-%d')
        overall_avg = int(actual_df_processed['y'].mean())
        # Use .iloc for positional access after confirming df is sorted and not empty
        start_val = int(actual_df_processed.iloc[0]['y'])
        end_val = int(actual_df_processed.iloc[-1]['y'])
        overall_trend = "Increased" if end_val > start_val else "Decreased" if end_val < start_val else "Remained Stable"

        historical_summary_str = f"""
        - Data Period: {start_date} to {end_date}
        - Starting Sessions: {start_val}
        - Ending Sessions: {end_val}
        - Average Sessions: {overall_avg}
        - Overall Trend During Period: {overall_trend}
        """

        updates_str = "\n".join([
            f"- {label} ({pd.to_datetime(start, format='%Y%m%d').strftime('%Y-%m-%d')} to {pd.to_datetime(end, format='%Y%m%d').strftime('%Y-%m-%d')})"
            for start, end, label in google_updates
        ])

        # --- Construct the prompt ---
        prompt = f"""
        Analyze the provided historical SEO traffic (sessions) data in the context of known Google Algorithm Updates.

        Context:
        - The data represents website sessions from Google Analytics.
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
         st.error("🔴 Gemini API Error: The prompt was blocked.")
         return "Analysis failed: Prompt was blocked by safety filters."
    except Exception as e:
        st.error(f"🔴 An error occurred while generating the AI historical analysis: {e}")
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
                     # Basic validation of format before full conversion
                     if not df['Date'].astype(str).str.match(r'^\d{8}$').all():
                          raise ValueError("Some 'Date' values are not in YYYYMMDD format.")
                     # We'll do the proper conversion later in plotting functions
                     # Just ensure Sessions is numeric
                     df['Sessions'] = pd.to_numeric(df['Sessions'], errors='coerce')
                     if df['Sessions'].isnull().any():
                         st.warning("⚠️ Warning: Some 'Sessions' values were non-numeric and have been ignored.")
                         df.dropna(subset=['Sessions'], inplace=True)
                     df['Sessions'] = df['Sessions'].astype(int) # Convert to int after cleaning NaNs
                     return df
                 except ValueError as ve:
                     st.error(f"🔴 Error: {ve}")
                     return None
                 except Exception as data_err:
                     st.error(f"🔴 Error processing CSV data columns: {data_err}")
                     return None
        except Exception as e:
            st.error(f"🔴 Error loading or parsing CSV: {e}")
            return None
    else:
        return None

# --- Plotting Functions (Revised Data Handling) ---
# They now EXPECT a DataFrame with 'Date' (YYYYMMDD string) and 'Sessions' (numeric)
# and will perform all necessary conversions/renaming/resampling internally.

def plot_daily_forecast(df_original, forecast_end_date, google_updates):
    """Generates daily forecast, plots actual vs forecast, returns results."""
    df = df_original.copy() # Work on a copy
    try:
        # --- Internal Data Prep ---
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
        df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
        df.sort_values('ds', inplace=True)
        # --- End Internal Data Prep ---
    except Exception as e:
        st.error(f"Error preparing data for daily plot: {e}")
        return None, None, None

    last_date = df['ds'].max()
    if df.empty or pd.isna(last_date):
         st.error("Input data empty or last date invalid after processing (daily).")
         return None, None, None
    last_actual_value = df.loc[df['ds'] == last_date, 'y'].iloc[0]

    periods = (forecast_end_date - last_date).days
    if periods <= 0:
        st.warning("Forecast end date not after last observed date. Plotting historical data only.")
        forecast = None # No forecast to generate
    else:
        try:
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=periods, freq='D')
            forecast = model.predict(future)
        except Exception as e:
            st.error(f"Error during Prophet forecasting (daily): {e}")
            return None, last_date, last_actual_value # Return historical info

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 8))
    try:
        ax.plot(df['ds'], df['y'], label='Actual', color='blue', marker='.', linestyle='-') # Add markers
        if forecast is not None:
            # Plot only future forecast points clearly separated
            forecast_future_part = forecast[forecast['ds'] > last_date]
            ax.plot(forecast_future_part['ds'], forecast_future_part['yhat'], label='Forecast', color='green', linestyle='--')
            # Optionally plot historical fit too
            # forecast_hist_part = forecast[forecast['ds'] <= last_date]
            # ax.plot(forecast_hist_part['ds'], forecast_hist_part['yhat'], label='Prophet Fit', color='orange', linestyle=':')

        # Plot Google updates
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2
                ax.axvspan(start_date, end_date, color='gray', alpha=0.2, label='_nolegend_') # Hide span from legend
                y_limits = ax.get_ylim()
                text_y_pos = y_limits[1] * 0.98 if y_limits and y_limits[1] > y_limits[0] else (df['y'].max() * 0.98)
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
        plt.close(fig)

    return forecast, last_date, last_actual_value


def plot_weekly_forecast(df_original, forecast_end_date, google_updates):
    """Generates weekly forecast, plots actual vs forecast, returns results."""
    df = df_original.copy()
    try:
        # --- Internal Data Prep ---
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
        df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
        df.sort_values('ds', inplace=True)
        df_weekly = df.set_index('ds').resample('W').sum().reset_index() # Resample
        # --- End Internal Data Prep ---
    except Exception as e:
        st.error(f"Error preparing data for weekly plot: {e}")
        return None, None, None

    if df_weekly.empty:
        st.error("No data after weekly resampling.")
        return None, None, None

    last_date = df_weekly['ds'].max()
    if pd.isna(last_date):
        st.error("Last date invalid after weekly resampling.")
        return None, None, None
    last_actual_value = df_weekly.loc[df_weekly['ds'] == last_date, 'y'].iloc[0]

    periods = math.ceil((forecast_end_date - last_date).days / 7)
    if periods <= 0:
        st.warning("Forecast end date not after last observed week. Plotting historical data only.")
        forecast = None
    else:
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
        ax.plot(df_weekly['ds'], df_weekly['y'], label='Weekly Actual', color='blue', marker='.', linestyle='-')
        if forecast is not None:
            forecast_future_part = forecast[forecast['ds'] > last_date]
            ax.plot(forecast_future_part['ds'], forecast_future_part['yhat'], label='Weekly Forecast', color='green', linestyle='--')

        # Plot Google updates
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2
                ax.axvspan(start_date, end_date, color='gray', alpha=0.2, label='_nolegend_')
                y_limits = ax.get_ylim()
                text_y_pos = y_limits[1] * 0.98 if y_limits and y_limits[1] > y_limits[0] else (df_weekly['y'].max() * 0.98)
                ax.text(mid_date, text_y_pos, label, ha='center', va='top', fontsize=8, rotation=90)
            except Exception as plot_err:
                 st.warning(f"Could not plot Google update '{label}': {plot_err}")

        ax.set_title('Weekly Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
        ax.set_xlabel('Date (Week Start)')
        ax.set_ylabel('Sessions (Weekly Total)')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during plotting (weekly): {e}")
    finally:
        plt.close(fig)

    return forecast, last_date, last_actual_value


def plot_monthly_forecast(df_original, forecast_end_date, google_updates):
    """Generates monthly forecast, plots actual vs forecast, returns results."""
    df = df_original.copy()
    try:
        # --- Internal Data Prep ---
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
        df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
        df.sort_values('ds', inplace=True)
        df_monthly = df.set_index('ds').resample('M').sum().reset_index() # Resample Month End
        # --- End Internal Data Prep ---
    except Exception as e:
        st.error(f"Error preparing data for monthly plot: {e}")
        return None, None, None

    if df_monthly.empty:
        st.error("No data after monthly resampling.")
        return None, None, None

    last_date = df_monthly['ds'].max()
    if pd.isna(last_date):
        st.error("Last date invalid after monthly resampling.")
        return None, None, None
    last_actual_value = df_monthly.loc[df_monthly['ds'] == last_date, 'y'].iloc[0]

    periods = 0
    temp_date = last_date
    while temp_date < forecast_end_date:
        temp_date += pd.offsets.MonthEnd(1)
        periods += 1

    if periods <= 0:
        st.warning("Forecast end date not after last observed month. Plotting historical data only.")
        forecast = None
    else:
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
        ax.plot(df_monthly['ds'], df_monthly['y'], label='Monthly Actual', color='blue', marker='.', linestyle='-')
        if forecast is not None:
            forecast_future_part = forecast[forecast['ds'] > last_date]
            ax.plot(forecast_future_part['ds'], forecast_future_part['yhat'], label='Monthly Forecast', color='green', linestyle='--')

        # Plot Google updates
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2
                ax.axvspan(start_date, end_date, color='gray', alpha=0.2, label='_nolegend_')
                y_limits = ax.get_ylim()
                text_y_pos = y_limits[1] * 0.98 if y_limits and y_limits[1] > y_limits[0] else (df_monthly['y'].max() * 0.98)
                ax.text(mid_date, text_y_pos, label, ha='center', va='top', fontsize=8, rotation=90)
            except Exception as plot_err:
                 st.warning(f"Could not plot Google update '{label}': {plot_err}")

        ax.set_title('Monthly Actual vs. Forecasted GA4 Sessions with Google Update Ranges')
        ax.set_xlabel('Date (Month End)')
        ax.set_ylabel('Sessions (Monthly Total)')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during plotting (monthly): {e}")
    finally:
        plt.close(fig)

    return forecast, last_date, last_actual_value

# --- display_dashboard remains the same (displays forecast results) ---
def display_dashboard(forecast, last_date, forecast_end_date, forecast_type_label):
    st.subheader("Forecast Data Table & Summary")
    # Filter forecast for dates strictly after the last actual date up to the chosen end date
    forecast_filtered = forecast[(forecast['ds'] > last_date) & (forecast['ds'] <= forecast_end_date)]

    if not forecast_filtered.empty:
        # Display rounded dataframe
        st.dataframe(forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].astype(
            {'yhat':'int', 'yhat_lower':'int', 'yhat_upper':'int'}
        ).reset_index(drop=True)) # Reset index for cleaner look
    else:
        st.write("No forecast data points fall within the selected future range.")

    horizon_str = "N/A"
    # Calculate horizon based on number of rows in the filtered future forecast
    if not forecast_filtered.empty:
        horizon = len(forecast_filtered)
        granularity = forecast_type_label.split(" ")[0] # Daily, Weekly, Monthly
        if granularity == "Daily": horizon_str = f"{horizon} days"
        elif granularity == "Weekly": horizon_str = f"{horizon} weeks"
        else: horizon_str = f"{horizon} months"
    elif (forecast_end_date > last_date): # Check if forecast period is valid but just empty
        horizon_str = "0 (within period)"
    else:
        horizon_str = "N/A (No forecast period)"


    # Display summary metrics using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Last Actual Date", value=f"{last_date.date() if last_date else 'N/A'}")
    with col2:
        st.metric(label="Forecast Horizon", value=f"{horizon_str}")

    # Display forecast value only if forecast is available and filtered data exists
    if not forecast_filtered.empty:
        forecast_value_at_end = forecast_filtered.iloc[-1]
        forecast_range = int(forecast_value_at_end['yhat_upper'] - forecast_value_at_end['yhat_lower'])
        delta_val = forecast_range / 2
        with col3:
            st.metric(label=f"Forecast at {forecast_value_at_end['ds'].date()}",
                      value=int(forecast_value_at_end['yhat']),
                      delta=f"±{delta_val:.0f} (Range: {forecast_range})",
                      delta_color="off")
    else:
         with col3:
            st.metric(label="Forecast Value", value="N/A")


# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Google Algorithm Impact Visualizer + AI Analysis", layout="wide")
    st.title("📈 Google Algorithm Impact Visualizer with AI Analysis")
    st.write("""
        Upload GA4 sessions CSV ('Date' as YYYYMMDD, 'Sessions'). Visualize historical trends against Google updates,
        generate a Prophet forecast, and get AI analysis of past performance related to updates.
    """)
    st.info("💡 Ensure CSV has 'Date' (YYYYMMDD format) and 'Sessions' (numeric) columns.")

    # --- Define Google Updates List ---
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
    # Use 'key' to help Streamlit manage state
    forecast_granularity = st.sidebar.radio("Select Forecast Granularity", ("Daily", "Weekly", "Monthly"), key="forecast_granularity_radio")
    default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
    min_date_allowed = pd.Timestamp.today().date() + timedelta(days=1) # Forecast must be future
    forecast_end_date_input = st.sidebar.date_input("Select Forecast End Date", value=default_forecast_end, min_value=min_date_allowed, key="forecast_date_input")
    forecast_end_date = pd.to_datetime(forecast_end_date_input)

    # --- File Upload ---
    df_original = load_data()

    # --- Main Processing Area ---
    if df_original is not None:
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df_original.head())

        # Prepare data specifically for historical AI analysis ('ds', 'y' format)
        df_processed_for_ai = None
        try:
            df_processed_for_ai = df_original.copy()
            df_processed_for_ai['Date'] = pd.to_datetime(df_processed_for_ai['Date'].astype(str), format='%Y%m%d')
            df_processed_for_ai.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)
            df_processed_for_ai.sort_values('ds', inplace=True, ignore_index=True)
        except Exception as e:
            st.error(f"Failed to prepare data for AI analysis: {e}")
            # Allow app to continue, but AI section will be disabled implicitly

        # --- AI Historical Analysis Section ---
        st.markdown("---")
        st.header("🤖 AI Historical Analysis vs. Google Updates")
        api_key_present = bool(os.getenv("GOOGLE_API_KEY"))

        if not api_key_present:
            st.warning("⚠️ GOOGLE_API_KEY environment variable not set. AI analysis disabled.")
        elif df_processed_for_ai is None:
             st.warning("⚠️ Data could not be processed for AI analysis.")
        else:
            # Only show button if API key is present AND data was processed
            if st.button("📊 Analyze Historical Traffic vs. Updates", key="analyze_historical_button"):
                if configure_gemini():
                    with st.spinner("🧠 Analyzing historical data with Gemini..."):
                        historical_analysis_result = get_gemini_historical_analysis(df_processed_for_ai, google_updates)
                    st.markdown(historical_analysis_result)
                # else: configure_gemini showed error

        # --- Forecasting and Plotting Section ---
        st.markdown("---")
        st.header("📊 Forecast & Visualization")

        plot_function = None
        if forecast_granularity == "Daily":
            plot_function = plot_daily_forecast
        elif forecast_granularity == "Weekly":
            plot_function = plot_weekly_forecast
        elif forecast_granularity == "Monthly":
            plot_function = plot_monthly_forecast

        if plot_function:
            try:
                # Pass a fresh COPY of the original data to the plotting function
                # It will handle its own prep (renaming, resampling etc.)
                with st.spinner(f"Generating {forecast_granularity} forecast and plot..."):
                    forecast_result = plot_function(df_original.copy(), forecast_end_date, google_updates)

                # Unpack results
                forecast, last_date, last_actual_value = None, None, None # Default values
                if forecast_result:
                     forecast, last_date, last_actual_value = forecast_result
                # else: plotting function might have shown error or returned Nones

                # Display Forecast Dashboard
                st.markdown("---") # Separator before dashboard
                if forecast is not None and last_date is not None:
                    display_dashboard(forecast, last_date, forecast_end_date, f"{forecast_granularity} Forecast")
                elif last_date is not None: # Plotting worked historically, but forecast failed/not requested
                    st.info("Forecast could not be generated (check forecast end date?). Displaying historical data summary only.")
                    # Show a minimal dashboard if forecast failed but we have last_date
                    display_dashboard(pd.DataFrame(), last_date, forecast_end_date, f"{forecast_granularity} Forecast") # Pass empty forecast df
                else: # Plotting function itself failed
                     st.error("Could not generate plot or forecast.")

            except Exception as e:
                st.error(f"🔴 An error occurred during the forecasting/plotting process: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")
        else:
            st.error("Internal error: Could not determine plotting function.")


    else:
        # Show message if no file uploaded
        if 'ga4_csv_uploader' not in st.session_state or st.session_state.ga4_csv_uploader is None:
              st.info("Awaiting CSV file upload...")

    # --- Footer ---
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")


if __name__ == "__main__":
    main()
