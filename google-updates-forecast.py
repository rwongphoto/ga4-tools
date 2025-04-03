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
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during Gemini configuration: {e}")
        return False

# --- NEW Gemini Function for Historical DEVIATION Analysis ---
def get_gemini_historical_deviation_analysis(historical_data_with_fit, google_updates):
    """
    Analyzes historical traffic deviations from Prophet's fit using Google Gemini,
    focusing on correlations with Google Algorithm Updates.

    Args:
        historical_data_with_fit (pd.DataFrame): DataFrame with historical data
                                                  ('ds', 'y' (actual), 'yhat' (predicted)).
        google_updates (list): List of tuples containing Google update info.

    Returns:
        str: The analysis summary from Gemini, or an error message.
    """
    if not configure_gemini():
        return "Gemini API not configured. Analysis cannot proceed."

    if historical_data_with_fit.empty:
        return "Historical data with fit is empty. Cannot perform analysis."

    # Calculate deviation
    historical_data_with_fit['deviation'] = historical_data_with_fit['y'] - historical_data_with_fit['yhat']
    historical_data_with_fit['deviation_pct'] = (historical_data_with_fit['deviation'] / historical_data_with_fit['yhat']) * 100
    historical_data_with_fit['deviation_pct'] = historical_data_with_fit['deviation_pct'].fillna(0).replace([float('inf'), -float('inf')], 0) # Handle potential division by zero


    try:
        # --- Prepare data summary for the prompt ---
        start_date = historical_data_with_fit['ds'].min().strftime('%Y-%m-%d')
        end_date = historical_data_with_fit['ds'].max().strftime('%Y-%m-%d')
        avg_deviation_pct = historical_data_with_fit['deviation_pct'].mean()

        # Find periods of large deviations (e.g., > 1 std deviation) - more advanced, maybe later
        # For now, just provide basic stats and let Gemini infer from the context.

        deviation_summary_str = f"""
        - Data Period Analyzed: {start_date} to {end_date}
        - Average Deviation from Expected (Prophet Fit): {avg_deviation_pct:.2f}%
        - Note: Positive deviation means actual traffic was higher than predicted by the model's baseline fit for that period; negative means lower.
        """

        # Format Google Updates for the prompt
        updates_str = "\n".join([
            f"- {label} ({pd.to_datetime(start, format='%Y%m%d').strftime('%Y-%m-%d')} to {pd.to_datetime(end, format='%Y%m%d').strftime('%Y-%m-%d')})"
            for start, end, label in google_updates
        ])

        # --- Construct the prompt ---
        prompt = f"""
        Analyze historical SEO traffic (sessions) data by comparing the ACTUAL traffic ('y') to the traffic PREDICTED by a Prophet time series model ('yhat') for the SAME historical period. The goal is to identify significant deviations and correlate them with Google Algorithm Updates.

        Context:
        - We are looking at the difference between actual performance and the baseline expectation set by the Prophet model's fit to the historical data.
        - Large positive deviations mean traffic over-performed the model's expectation.
        - Large negative deviations mean traffic under-performed the model's expectation.

        Historical Deviation Summary:
        {deviation_summary_str}

        Google Algorithm Updates During Data Period (YYYY-MM-DD):
        {updates_str}

        Task:
        Provide a concise analysis (around 3-5 bullet points or a short paragraph) summarizing the relationship between historical traffic *deviations* (actual vs. predicted) and the timing of Google updates. Focus on:
        1. Identifying periods where actual traffic significantly deviated (positively or negatively) from the Prophet model's prediction ('yhat').
        2. Explicitly stating whether these significant deviation periods coincide with (occur during or shortly after) specific Google update periods listed.
        3. Noting any Google updates that appear to correlate strongly with either positive or negative deviations from the expected trend.
        4. Mentioning updates that seem to have had little impact on the *deviation* (i.e., traffic performed roughly as the model expected during those times).
        5. Frame the analysis for an SEO Manager assessing the *impact* of past updates relative to the expected baseline trend. Do NOT analyze the future forecast itself.
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
        st.error(f"🔴 An error occurred while generating the AI deviation analysis: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return f"Analysis failed due to an error: {e}"


# --- load_data function remains the same ---
def load_data():
    uploaded_file = st.file_uploader(
        "Choose a GA4 CSV file", type="csv", key="ga4_csv_uploader"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("🔴 Error: CSV must contain 'Date' and 'Sessions' columns.")
                return None
            # Perform validation and basic type conversion here
            try:
                if not df['Date'].astype(str).str.match(r'^\d{8}$').all():
                    raise ValueError("Some 'Date' values are not in YYYYMMDD format.")
                df['Sessions'] = pd.to_numeric(df['Sessions'], errors='coerce')
                if df['Sessions'].isnull().any():
                    st.warning("⚠️ Non-numeric 'Sessions' values found and ignored.")
                    df.dropna(subset=['Sessions'], inplace=True)
                df['Sessions'] = df['Sessions'].astype(int)
                # Convert Date to datetime object here for consistency
                df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
                df.sort_values('Date', inplace=True, ignore_index=True) # Sort early
                return df
            except ValueError as ve:
                st.error(f"🔴 Error validating data: {ve}")
                return None
            except Exception as data_err:
                st.error(f"🔴 Error processing CSV data columns: {data_err}")
                return None
        except Exception as e:
            st.error(f"🔴 Error loading or parsing CSV: {e}")
            return None
    return None # No file uploaded


# --- MODIFIED Plotting Functions ---
# Now handle fitting, predicting historical + future, and plotting all components.
# They return the FULL forecast DataFrame (history+future) and the historical part with actuals.

def run_prophet_and_plot(df_original, forecast_end_date, google_updates, granularity):
    """
    Fits Prophet, predicts historical fit & future forecast, plots results.

    Args:
        df_original (pd.DataFrame): DF with 'Date' (datetime) and 'Sessions' columns.
        forecast_end_date (pd.Timestamp): User selected end date for future forecast.
        google_updates (list): List of Google update tuples.
        granularity (str): 'Daily', 'Weekly', or 'Monthly'.

    Returns:
        tuple: (full_forecast_df, historical_data_with_fit, last_actual_date)
               Returns (None, None, None) on failure.
    """
    df = df_original.copy()
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True) # Rename now

    # --- Handle Granularity ---
    df_prophet = df # Default is Daily
    resample_freq = 'D'
    if granularity == 'Weekly':
        resample_freq = 'W'
        df_prophet = df.set_index('ds').resample(resample_freq).sum().reset_index()
    elif granularity == 'Monthly':
        resample_freq = 'M'
        df_prophet = df.set_index('ds').resample(resample_freq).sum().reset_index()

    if df_prophet.empty:
        st.error(f"Data empty after resampling to {granularity}.")
        return None, None, None

    last_actual_date = df_prophet['ds'].max()
    if pd.isna(last_actual_date):
        st.error(f"Could not determine last actual date for {granularity} data.")
        return None, None, None

    # --- Prophet Forecasting ---
    try:
        model = Prophet()
        model.fit(df_prophet)

        # Create dataframe for prediction (historical dates + future dates)
        periods = 0
        if forecast_end_date > last_actual_date:
             # Calculate periods based on granularity
             if granularity == 'Daily':
                 periods = (forecast_end_date - last_actual_date).days
             elif granularity == 'Weekly':
                 periods = math.ceil((forecast_end_date - last_actual_date).days / 7)
             elif granularity == 'Monthly':
                  temp_date = last_actual_date
                  while temp_date < forecast_end_date:
                      temp_date += pd.offsets.MonthEnd(1)
                      periods += 1

        # Make future dataframe including history AND future periods
        # This ensures Prophet predicts 'yhat' for historical dates too
        future_df = model.make_future_dataframe(periods=periods, freq=resample_freq, include_history=True)

        # Predict
        full_forecast_df = model.predict(future_df)

    except Exception as e:
        st.error(f"Error during Prophet modeling/prediction ({granularity}): {e}")
        st.error(traceback.format_exc())
        return None, None, last_actual_date # Return last date if possible

    # --- Prepare historical data with fit for deviation analysis ---
    # Merge prophet's historical predictions back with actuals
    historical_data_with_fit = pd.merge(
        df_prophet, # Actual values (resampled if needed)
        full_forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], # Predictions
        on='ds',
        how='left' # Keep all actual dates
    )

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(16, 8))
    try:
        # 1. Plot Actual Historical Data
        ax.plot(historical_data_with_fit['ds'], historical_data_with_fit['y'],
                label='Actual (' + granularity + ')', color='blue', marker='.', markersize=4, linestyle='-')

        # 2. Plot Prophet's Prediction/Fit for the historical period
        ax.plot(historical_data_with_fit['ds'], historical_data_with_fit['yhat'],
                label='Prophet Predicted Fit', color='orange', linestyle=':')

        # 3. Plot Future Forecast (if exists)
        future_forecast_part = full_forecast_df[full_forecast_df['ds'] > last_actual_date]
        if not future_forecast_part.empty:
            ax.plot(future_forecast_part['ds'], future_forecast_part['yhat'],
                    label='Prophet Future Forecast', color='green', linestyle='--')
            # 4. Plot Confidence Interval (only for future part for clarity, or full?)
            # Plotting full interval:
            ax.fill_between(full_forecast_df['ds'], full_forecast_df['yhat_lower'], full_forecast_df['yhat_upper'],
                           color='skyblue', alpha=0.3, label='Confidence Interval')


        # 5. Plot Google Updates
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2
                ax.axvspan(start_date, end_date, color='gray', alpha=0.2, label='_nolegend_')
                y_limits = ax.get_ylim()
                # Adjust y-position calculation based on actual data max
                text_y_pos = (y_limits[1] * 0.98) if y_limits and y_limits[1] > y_limits[0] else (historical_data_with_fit['y'].max() * 1.02) # Place slightly above max actual
                ax.text(mid_date, text_y_pos, label, ha='center', va='top', fontsize=8, rotation=90)
            except Exception as plot_err:
                 st.warning(f"Could not plot Google update '{label}': {plot_err}")

        ax.set_title(f'{granularity} Actual vs. Prophet Fit & Forecast with Google Updates')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sessions')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during plotting ({granularity}): {e}")
    finally:
        plt.close(fig)

    return full_forecast_df, historical_data_with_fit, last_actual_date


# --- display_dashboard remains similar (shows FUTURE forecast summary) ---
def display_dashboard(full_forecast_df, last_actual_date, forecast_end_date, granularity_label):
    st.subheader(f"Future Forecast Summary ({granularity_label})")

    # Filter for future dates only
    forecast_future = full_forecast_df[
        (full_forecast_df['ds'] > last_actual_date) &
        (full_forecast_df['ds'] <= forecast_end_date)
    ].copy() # Ensure it's a copy

    if not forecast_future.empty:
        # Display table of future points
        st.dataframe(forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].astype(
            {'yhat':'int', 'yhat_lower':'int', 'yhat_upper':'int'}
        ).reset_index(drop=True))

        horizon = len(forecast_future)
        unit = granularity_label.lower().replace("ly","") # day, week, month
        horizon_str = f"{horizon} {unit}{'s' if horizon != 1 else ''}"

        # Display summary metrics using columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Last Actual Date", value=f"{last_actual_date.date() if last_actual_date else 'N/A'}")
        with col2:
            st.metric(label="Forecast Horizon", value=f"{horizon_str}")

        forecast_value_at_end = forecast_future.iloc[-1]
        forecast_range = int(forecast_value_at_end['yhat_upper'] - forecast_value_at_end['yhat_lower'])
        delta_val = forecast_range / 2
        with col3:
            st.metric(label=f"Forecast at {forecast_value_at_end['ds'].date()}",
                      value=int(forecast_value_at_end['yhat']),
                      delta=f"±{delta_val:.0f} (Range: {forecast_range})",
                      delta_color="off")
    else:
        st.info("No future forecast points fall within the selected date range or forecast was not generated.")
        # Still show last actual date if available
        st.metric(label="Last Actual Date", value=f"{last_actual_date.date() if last_actual_date else 'N/A'}")


# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Google Algorithm Impact Analyzer", layout="wide") # Updated Title
    st.title("📈 Google Algorithm Impact Analyzer")
    st.write("""
        Upload GA4 sessions CSV ('Date' as YYYYMMDD, 'Sessions').
        Visualize **Actual Traffic** vs. **Prophet's Predicted Fit** against Google updates.
        Optionally generate a future forecast and get AI analysis of **historical deviations** correlated with updates.
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
    granularity = st.sidebar.radio("Select Analysis Granularity", ("Daily", "Weekly", "Monthly"), key="granularity_radio")
    # Option to enable/disable future forecast - default ON
    show_future_forecast = st.sidebar.checkbox("Include Future Forecast?", value=True, key="show_future_cb")
    forecast_end_date = None
    if show_future_forecast:
        default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
        min_date_allowed = pd.Timestamp.today().date() + timedelta(days=1)
        forecast_end_date_input = st.sidebar.date_input(
            "Select Forecast End Date",
            value=default_forecast_end,
            min_value=min_date_allowed,
            key="forecast_date_input",
            disabled=not show_future_forecast # Disable if checkbox is off
        )
        forecast_end_date = pd.to_datetime(forecast_end_date_input)
    else:
        # Set forecast_end_date to last possible date if forecast disabled,
        # to prevent errors in period calculation, Prophet will ignore it.
        # Or more simply, pass None and handle it in run_prophet_and_plot
        forecast_end_date = pd.Timestamp.today() # Needs a value, even if unused

    # --- File Upload ---
    df_original = load_data() # Now returns df with 'Date' as datetime object

    # --- Main Processing Area ---
    if df_original is not None:
        st.subheader("Data Preview (First 5 Rows)")
        # Show Date in YYYY-MM-DD format for preview
        st.dataframe(df_original.head().assign(Date=lambda x: x['Date'].dt.strftime('%Y-%m-%d')))

        # --- Run Prophet & Plotting Section ---
        st.markdown("---")
        st.header("📊 Historical Performance vs. Prophet Fit & Forecast")

        full_forecast_df = None
        historical_data_with_fit = None
        last_actual_date = None

        try:
            # Determine effective end date for Prophet run
            # If not showing future forecast, set end date to last actual date
            effective_end_date = forecast_end_date if show_future_forecast else df_original['Date'].max()

            with st.spinner(f"Running Prophet ({granularity}) and generating plot..."):
                results = run_prophet_and_plot(
                    df_original.copy(),
                    effective_end_date, # Pass effective end date
                    google_updates,
                    granularity
                )

            if results:
                full_forecast_df, historical_data_with_fit, last_actual_date = results
            # else: Error message was shown in the function

        except Exception as e:
            st.error(f"🔴 An error occurred during the main analysis process: {e}")
            st.error(f"Traceback: {traceback.format_exc()}")


        # --- AI Historical DEVIATION Analysis Section ---
        st.markdown("---")
        st.header("🤖 AI Analysis: Historical Deviations vs. Google Updates")
        api_key_present = bool(os.getenv("GOOGLE_API_KEY"))

        if not api_key_present:
            st.warning("⚠️ GOOGLE_API_KEY environment variable not set. AI analysis disabled.")
        elif historical_data_with_fit is None or historical_data_with_fit.empty:
             st.warning("⚠️ Cannot perform AI analysis: Historical data with Prophet fit not available.")
        else:
            # Only show button if API key is present AND historical fit data exists
            if st.button("📈 Analyze Historical Performance Deviations", key="analyze_deviation_button"):
                if configure_gemini():
                    with st.spinner("🧠 Analyzing historical deviations with Gemini..."):
                        deviation_analysis_result = get_gemini_historical_deviation_analysis(
                            historical_data_with_fit.copy(), # Pass a copy
                            google_updates
                        )
                    st.markdown(deviation_analysis_result)
                # else: configure_gemini showed error


        # --- Display Future Forecast Dashboard (if applicable) ---
        st.markdown("---")
        if show_future_forecast and full_forecast_df is not None and last_actual_date is not None:
             display_dashboard(full_forecast_df, last_actual_date, forecast_end_date, granularity)
        elif show_future_forecast:
            st.info("Future forecast dashboard could not be displayed (forecasting might have failed or data was insufficient).")


    else:
        # Show message if no file uploaded
        if 'ga4_csv_uploader' not in st.session_state or st.session_state.ga4_csv_uploader is None:
              st.info("Awaiting CSV file upload...")

    # --- Footer ---
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")


if __name__ == "__main__":
    main()
