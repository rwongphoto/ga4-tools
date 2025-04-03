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
        # Avoid showing error directly here, let the calling function handle UI
        # st.error("🔴 Error: GOOGLE_API_KEY environment variable not found or empty.")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during Gemini configuration: {e}")
        return False

# --- Gemini Function for Historical DEVIATION Analysis ---
def get_gemini_historical_deviation_analysis(historical_data_with_fit, google_updates):
    """
    Analyzes historical traffic deviations from Prophet's fit using Google Gemini,
    focusing on correlations with Google Algorithm Updates (positive, negative, or none).

    Args:
        historical_data_with_fit (pd.DataFrame): DataFrame with historical data
                                                  ('ds', 'y' (actual), 'yhat' (predicted)).
        google_updates (list): List of tuples containing Google update info.

    Returns:
        str: The analysis summary from Gemini, or an error message.
    """
    # Check configuration status before proceeding
    if not configure_gemini():
        st.error("🔴 Error: GOOGLE_API_KEY environment variable not found or empty. Cannot perform AI analysis.")
        return "Gemini API not configured. Analysis cannot proceed."

    if historical_data_with_fit is None or historical_data_with_fit.empty:
        return "Historical data with fit is empty. Cannot perform analysis."

    # Calculate deviation (ensure working on a copy)
    analysis_df = historical_data_with_fit.copy()
    analysis_df['deviation'] = analysis_df['y'] - analysis_df['yhat']
    # Calculate pct deviation robustly to avoid division by zero or near-zero
    analysis_df['deviation_pct'] = analysis_df.apply(
        lambda row: (row['deviation'] / row['yhat'] * 100) if row['yhat'] is not None and abs(row['yhat']) > 1e-6 else 0, axis=1 # Avoid division by zero/small numbers
    )
    # Handle potential NaNs or Infs arising from calculation if any edge cases remain
    analysis_df['deviation_pct'] = analysis_df['deviation_pct'].fillna(0).replace([float('inf'), -float('inf')], 0)


    try:
        # --- Prepare data summary for the prompt ---
        start_date = analysis_df['ds'].min().strftime('%Y-%m-%d')
        end_date = analysis_df['ds'].max().strftime('%Y-%m-%d')
        avg_deviation_pct = analysis_df['deviation_pct'].mean()

        deviation_summary_str = f"""
        - Data Period Analyzed: {start_date} to {end_date}
        - Average Deviation from Expected (Prophet Fit): {avg_deviation_pct:.2f}%
        - Note: Positive deviation means actual traffic was higher than predicted by the model's baseline fit; negative means lower.
        """

        # Format Google Updates for the prompt
        updates_str = "\n".join([
            f"- {label} ({pd.to_datetime(start, format='%Y%m%d').strftime('%Y-%m-%d')} to {pd.to_datetime(end, format='%Y%m%d').strftime('%Y-%m-%d')})"
            for start, end, label in google_updates
        ])

        # --- Construct the prompt ---
        prompt = f"""
        Analyze historical SEO traffic (sessions) data by comparing the ACTUAL traffic ('y') to the traffic PREDICTED by a Prophet time series model ('yhat') for the SAME historical period. The goal is to identify significant deviations (periods where actual performance differed notably from the model's expectation) and correlate them with Google Algorithm Updates.

        Context:
        - We are looking at the difference between actual performance and the baseline expectation set by the Prophet model's fit to the historical data.
        - Large positive deviations (actual >> predicted) mean traffic over-performed the model's expectation. This could indicate a positive impact from external factors like a Google update, or successful SEO efforts.
        - Large negative deviations (actual << predicted) mean traffic under-performed the model's expectation. This could indicate a negative impact from external factors like a Google update, technical issues, or increased competition.

        Historical Deviation Summary:
        {deviation_summary_str}

        Google Algorithm Updates During Data Period (YYYY-MM-DD):
        {updates_str}

        Task:
        Provide a concise analysis (around 3-5 bullet points or a short paragraph) summarizing the relationship between historical traffic *deviations* (actual vs. predicted) and the timing of Google updates. Focus specifically on:
        1. Identifying periods where actual traffic significantly deviated (both positively and negatively) from the Prophet model's prediction ('yhat').
        2. Explicitly stating whether these significant deviation periods coincide with (occur during or shortly after) specific Google update periods listed.
        3. Highlighting any Google updates that appear to correlate strongly with **positive deviations** (better-than-expected performance).
        4. Highlighting any Google updates that appear to correlate strongly with **negative deviations** (worse-than-expected performance).
        5. Mentioning updates that seem to have had little impact on the *deviation* (i.e., traffic performed roughly as the model expected during those times, regardless of the overall trend).
        6. Frame the analysis for an SEO Manager assessing the *impact* of past updates relative to the expected baseline trend. Do NOT analyze the future forecast itself.
        """

        # --- Call the Gemini API ---
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        analysis = response.text.replace('•', '*') # Basic markdown cleanup
        return analysis

    except genai.types.generation_types.BlockedPromptException:
         st.error("🔴 Gemini API Error: The prompt was blocked due to safety settings.")
         return "Analysis failed: Prompt was blocked by safety filters."
    except Exception as e:
        st.error(f"🔴 An error occurred while generating the AI deviation analysis: {e}")
        # st.error(f"Traceback: {traceback.format_exc()}") # Optional: Show full traceback in UI for debugging
        return f"Analysis failed due to an error: {e}"

# --- Data Loading Function ---
def load_data():
    """Handles file upload and initial data validation/processing."""
    uploaded_file = st.file_uploader(
        "Choose a GA4 CSV file", type="csv", key="ga4_csv_uploader"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Basic Column Check
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("🔴 Error: CSV must contain 'Date' and 'Sessions' columns.")
                return None

            # Data Type and Format Validation
            try:
                # Ensure Date is string before regex check
                date_str_series = df['Date'].astype(str)
                if not date_str_series.str.match(r'^\d{8}$').all():
                    raise ValueError("Some 'Date' values are not in YYYYMMDD format.")

                # Convert Sessions to numeric, handle errors
                df['Sessions'] = pd.to_numeric(df['Sessions'], errors='coerce')
                if df['Sessions'].isnull().any():
                    st.warning("⚠️ Non-numeric 'Sessions' values found and ignored.")
                    df.dropna(subset=['Sessions'], inplace=True) # Remove rows with invalid sessions

                # Check if DataFrame became empty after dropping NaNs
                if df.empty:
                     st.error("🔴 Error: No valid numeric 'Sessions' data found.")
                     return None

                df['Sessions'] = df['Sessions'].astype(int) # Convert valid sessions to int

                # Convert Date to datetime objects *after* validation
                df['Date'] = pd.to_datetime(date_str_series, format='%Y%m%d')

                # Sort by date (important!)
                df.sort_values('Date', inplace=True, ignore_index=True)
                return df

            except ValueError as ve:
                st.error(f"🔴 Error validating data format: {ve}")
                return None
            except Exception as data_err:
                st.error(f"🔴 Error processing CSV data columns: {data_err}")
                return None
        except Exception as e:
            st.error(f"🔴 Error loading or parsing CSV: {e}")
            return None
    return None # No file uploaded

# --- Prophet Modeling and Plotting Function ---
def run_prophet_and_plot(df_original, effective_end_date, google_updates, granularity):
    """
    Fits Prophet, predicts historical fit & future forecast, plots results.

    Args:
        df_original (pd.DataFrame): DF with 'Date' (datetime) and 'Sessions'.
        effective_end_date (pd.Timestamp): Furthest date for Prophet prediction (can be last actual date or future date).
        google_updates (list): List of Google update tuples.
        granularity (str): 'Daily', 'Weekly', or 'Monthly'.

    Returns:
        tuple: (full_forecast_df, historical_data_with_fit, last_actual_date)
               Returns (None, None, None) on failure.
    """
    df = df_original.copy()
    # Rename columns for Prophet
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)

    # --- Handle Granularity & Resampling ---
    df_prophet = df # Default is Daily
    resample_freq = 'D'
    if granularity == 'Weekly':
        resample_freq = 'W'
        try:
            # Use 'W-SUN' or other appropriate week start if needed
            df_prophet = df.set_index('ds').resample(resample_freq).sum().reset_index()
        except Exception as e:
             st.error(f"Error during weekly resampling: {e}")
             return None, None, None
    elif granularity == 'Monthly':
        resample_freq = 'M' # Month End frequency
        try:
            df_prophet = df.set_index('ds').resample(resample_freq).sum().reset_index()
        except Exception as e:
             st.error(f"Error during monthly resampling: {e}")
             return None, None, None

    if df_prophet.empty:
        st.error(f"Data empty after resampling to {granularity}.")
        return None, None, None

    last_actual_date = df_prophet['ds'].max()
    if pd.isna(last_actual_date):
        st.error(f"Could not determine last actual date for {granularity} data.")
        return None, None, None

    # --- Prophet Forecasting ---
    full_forecast_df = None
    try:
        # Check if effective_end_date is valid relative to last_actual_date
        if effective_end_date < last_actual_date:
            st.warning(f"Effective end date ({effective_end_date.date()}) is before last actual date ({last_actual_date.date()}). Analyzing historical fit only.")
            effective_end_date = last_actual_date # Limit prediction to historical data

        # Calculate future periods needed AFTER last actual date
        periods = 0
        if effective_end_date > last_actual_date:
             if granularity == 'Daily':
                 periods = (effective_end_date - last_actual_date).days
             elif granularity == 'Weekly':
                 # Calculate weeks based on the end date relative to the last week's end date
                 periods = math.ceil((effective_end_date - last_actual_date).days / 7)
             elif granularity == 'Monthly':
                  temp_date = last_actual_date
                  while temp_date < effective_end_date:
                      # Move to the next month end
                      temp_date += pd.offsets.MonthEnd(1)
                      # Only count if the new month end is within the effective end date
                      if temp_date <= effective_end_date:
                          periods += 1

        model = Prophet()
        # Add seasonality explicitly if needed, especially for weekly/monthly
        if granularity == 'Weekly':
            model.add_seasonality(name='yearly', period=365.25/7, fourier_order=5) # Example weekly seasonality
        if granularity == 'Monthly':
             model.add_seasonality(name='yearly', period=12, fourier_order=5, mode='multiplicative') # Example monthly seasonality


        model.fit(df_prophet)

        # Make dataframe for prediction (includes history + future periods)
        future_df = model.make_future_dataframe(periods=periods, freq=resample_freq, include_history=True)

        # Predict
        full_forecast_df = model.predict(future_df)

    except Exception as e:
        st.error(f"Error during Prophet modeling/prediction ({granularity}): {e}")
        st.error(traceback.format_exc())
        # Attempt to return historical data even if forecast fails
        historical_data_with_fit_on_error = pd.merge(
             df_prophet, pd.DataFrame({'ds': df_prophet['ds']}), on='ds', how='left'
        ).assign(yhat=None, yhat_lower=None, yhat_upper=None) # Add empty columns
        return None, historical_data_with_fit_on_error, last_actual_date


    # --- Prepare historical data with fit ---
    # Merge prophet's predictions back with actuals for the historical part
    historical_data_with_fit = pd.merge(
        df_prophet, # Actual values (resampled if needed)
        full_forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], # Predictions
        on='ds',
        how='left' # Keep all actual dates, even if Prophet didn't predict for some edge reason
    )

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(16, 9)) # Slightly taller figure
    try:
        # 1. Plot Actual Historical Data
        ax.plot(historical_data_with_fit['ds'], historical_data_with_fit['y'],
                label=f'Actual ({granularity})', color='blue', marker='.', markersize=5, linestyle='-', zorder=3)

        # 2. Plot Prophet's Prediction/Fit for the historical period
        ax.plot(historical_data_with_fit['ds'], historical_data_with_fit['yhat'],
                label='Prophet Predicted Fit (Historical)', color='orange', linestyle=':', linewidth=1.5, zorder=2)

        # 3. Plot Future Forecast (if available in full_forecast_df)
        future_forecast_part = full_forecast_df[full_forecast_df['ds'] > last_actual_date]
        if not future_forecast_part.empty:
            ax.plot(future_forecast_part['ds'], future_forecast_part['yhat'],
                    label='Prophet Future Forecast', color='green', linestyle='--', linewidth=2, zorder=2)

        # 4. Plot Confidence Interval (entire range for context)
        ax.fill_between(full_forecast_df['ds'], full_forecast_df['yhat_lower'], full_forecast_df['yhat_upper'],
                       color='skyblue', alpha=0.3, label='Confidence Interval (80%)', zorder=1)


        # 5. Plot Google Updates
        # Keep track of labels added to avoid duplicates in legend
        update_labels_added = set()
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2

                # Check if update range overlaps with data range at all
                if start_date <= historical_data_with_fit['ds'].max() and end_date >= historical_data_with_fit['ds'].min():
                    span_label = 'Google Update Period' if label not in update_labels_added else '_nolegend_'
                    ax.axvspan(start_date, end_date, color='lightcoral', alpha=0.25, label=span_label, zorder=0)
                    update_labels_added.add(label) # Mark as added

                    # Add text label for the update
                    y_limits = ax.get_ylim()
                    # Place text slightly above the max *actual* value within the plot's current y-limits
                    y_max_in_view = historical_data_with_fit['y'].max() # Consider max actual for positioning text
                    text_y_pos = (y_limits[1] * 1.02) if y_limits and y_limits[1] > y_limits[0] else (y_max_in_view * 1.05) # Default slightly above max actual
                    ax.text(mid_date, text_y_pos, label, ha='center', va='bottom', fontsize=7, rotation=90, color='dimgray', zorder=4) # Place below top edge

            except Exception as plot_err:
                 st.warning(f"Could not plot Google update '{label}': {plot_err}")

        ax.set_title(f'{granularity} Actual vs. Prophet Fit & Forecast with Google Updates')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sessions')
        # Improve legend placement
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout() # Adjust layout
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during plotting ({granularity}): {e}")
    finally:
        plt.close(fig) # Ensure figure is closed to free memory

    return full_forecast_df, historical_data_with_fit, last_actual_date


# --- Future Forecast Dashboard Display ---
def display_dashboard(full_forecast_df, last_actual_date, forecast_end_date, granularity_label):
    """Displays table and summary for the FUTURE forecast part."""
    st.subheader(f"Future Forecast Summary ({granularity_label})")

    # Filter for future dates only (strictly after last actual) up to user-specified end
    forecast_future = full_forecast_df[
        (full_forecast_df['ds'] > last_actual_date) &
        (full_forecast_df['ds'] <= forecast_end_date)
    ].copy() # Ensure it's a copy

    if not forecast_future.empty:
        # Display table of future points
        st.write("Forecasted Values:")
        st.dataframe(forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].astype(
            {'yhat':'int', 'yhat_lower':'int', 'yhat_upper':'int'}
        ).reset_index(drop=True), height=300) # Set height for scrollable table

        horizon = len(forecast_future)
        unit_map = {'Daily': 'day', 'Weekly': 'week', 'Monthly': 'month'}
        unit = unit_map.get(granularity_label, 'period')
        horizon_str = f"{horizon} {unit}{'s' if horizon != 1 else ''}"

        # Display summary metrics using columns
        st.write("Summary Metrics:")
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
        st.info("No future forecast points fall within the selected date range.")
        # Still show last actual date if available
        st.metric(label="Last Actual Date", value=f"{last_actual_date.date() if last_actual_date else 'N/A'}")


# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Google Algorithm Impact Analyzer", layout="wide")
    st.title("📈 Google Algorithm Impact Analyzer")
    st.write("""
        Upload GA4 sessions CSV ('Date' as YYYYMMDD, 'Sessions').
        Visualize **Actual Traffic** vs. **Prophet's Predicted Fit** for the historical period, overlaid with Google updates.
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
    show_future_forecast = st.sidebar.checkbox("Include Future Forecast?", value=True, key="show_future_cb")

    forecast_end_date = None # Initialize
    user_selected_forecast_end_date = None # Store the user's choice if forecast is on

    if show_future_forecast:
        default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
        # Ensure min_date is at least one day after today
        min_date_allowed = max(pd.Timestamp.today().date() + timedelta(days=1), default_forecast_end) # Ensure default is valid too if today+90 is past
        # Allow user to select date only if checkbox is ticked
        forecast_end_date_input = st.sidebar.date_input(
            "Select Forecast End Date",
            value=default_forecast_end,
            min_value=min_date_allowed,
            key="forecast_date_input",
            disabled=not show_future_forecast
        )
        user_selected_forecast_end_date = pd.to_datetime(forecast_end_date_input)


    # --- File Upload ---
    df_original = load_data() # Returns df with 'Date' as datetime

    # --- Main Processing Area ---
    if df_original is not None:
        st.subheader("Data Preview (First 5 Rows)")
        # Show Date in YYYY-MM-DD format for preview
        st.dataframe(df_original.head().assign(Date=lambda x: x['Date'].dt.strftime('%Y-%m-%d')))

        # --- Run Prophet & Plotting ---
        st.markdown("---")
        st.header("📊 Historical Performance vs. Prophet Fit & Forecast")

        full_forecast_df = None
        historical_data_with_fit = None
        last_actual_date = None

        # Determine the effective end date for the Prophet run
        # If forecast is off, predict only up to the last actual date
        # If forecast is on, use the user-selected future date
        effective_end_date_for_prophet = user_selected_forecast_end_date if show_future_forecast else df_original['Date'].max()

        try:
            with st.spinner(f"Running Prophet ({granularity}) and generating plot..."):
                results = run_prophet_and_plot(
                    df_original.copy(),
                    effective_end_date_for_prophet, # Use the determined end date
                    google_updates,
                    granularity
                )

            if results:
                full_forecast_df, historical_data_with_fit, last_actual_date = results
            # else: Error message likely shown within the function

        except Exception as e:
            st.error(f"🔴 An error occurred during the main analysis process: {e}")
            st.error(f"Traceback: {traceback.format_exc()}")


        # --- AI Historical DEVIATION Analysis Section ---
        st.markdown("---")
        st.header("🤖 AI Analysis: Historical Deviations vs. Google Updates")
        api_key_present = bool(os.getenv("GOOGLE_API_KEY"))

        if not api_key_present:
            st.warning("⚠️ GOOGLE_API_KEY environment variable not set. AI analysis disabled.")
        # Check if historical fit data is available for analysis
        elif historical_data_with_fit is None or historical_data_with_fit.empty:
             st.warning("⚠️ Cannot perform AI analysis: Historical data with Prophet fit not available (Prophet modeling might have failed).")
        else:
            # Only show button if API key is present AND historical fit data exists
            if st.button("📈 Analyze Historical Performance Deviations", key="analyze_deviation_button"):
                # Check config right before call for robustness
                if configure_gemini():
                    with st.spinner("🧠 Analyzing historical deviations with Gemini..."):
                        deviation_analysis_result = get_gemini_historical_deviation_analysis(
                            historical_data_with_fit.copy(), # Pass a copy just in case
                            google_updates
                        )
                    st.markdown(deviation_analysis_result)
                # else: configure_gemini would have shown an error if it failed

        # --- Display Future Forecast Dashboard (if requested and successful) ---
        st.markdown("---") # Separator before optional dashboard
        if show_future_forecast:
            if full_forecast_df is not None and last_actual_date is not None:
                 # Pass the user selected end date for filtering the dashboard view
                 display_dashboard(full_forecast_df, last_actual_date, user_selected_forecast_end_date, granularity)
            else:
                st.info("Future forecast could not be generated or displayed.")
        else:
            st.caption("Future forecast display disabled via sidebar option.")


    else:
        # Show message if no file uploaded
        # Check session state for the uploader key to avoid showing after upload attempt fails
        uploader_state = st.session_state.get("ga4_csv_uploader")
        if uploader_state is None:
              st.info("Awaiting CSV file upload...")

    # --- Footer ---
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")


if __name__ == "__main__":
    main()
