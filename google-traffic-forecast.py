import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import math
from datetime import timedelta
import google.generativeai as genai
import traceback
import os
import holoviews as hv
hv.extension('bokeh')
from st_bokeh import st_bokeh # <-- Import the custom component

# --- Google Gemini Configuration ---
def configure_gemini():
    """Configures the Gemini API using the GOOGLE_API_KEY environment variable."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
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
    if not configure_gemini():
        st.error("🔴 Error: GOOGLE_API_KEY environment variable not found or empty. Cannot perform AI analysis.")
        return "Gemini API not configured. Analysis cannot proceed."

    if historical_data_with_fit is None or historical_data_with_fit.empty:
        return "Historical data with fit is empty. Cannot perform analysis."

    analysis_df = historical_data_with_fit.copy()
    analysis_df['deviation'] = analysis_df['y'] - analysis_df['yhat']
    analysis_df['deviation_pct'] = analysis_df.apply(
        lambda row: (row['deviation'] / row['yhat'] * 100) if row['yhat'] is not None and abs(row['yhat']) > 1e-6 else 0, axis=1
    )
    analysis_df['deviation_pct'] = analysis_df['deviation_pct'].fillna(0).replace([float('inf'), -float('inf')], 0)

    try:
        start_date = analysis_df['ds'].min().strftime('%Y-%m-%d')
        end_date = analysis_df['ds'].max().strftime('%Y-%m-%d')
        avg_deviation_pct = analysis_df['deviation_pct'].mean()

        deviation_summary_str = f"""
        - Data Period Analyzed: {start_date} to {end_date}
        - Average Deviation from Expected (Prophet Fit): {avg_deviation_pct:.2f}%
        - Note: Positive deviation means actual traffic was higher than predicted by the model's baseline fit; negative means lower.
        """

        updates_str = "\n".join([
            f"- {label} ({pd.to_datetime(start, format='%Y%m%d').strftime('%Y-%m-%d')} to {pd.to_datetime(end, format='%Y%m%d').strftime('%Y-%m-%d')})"
            for start, end, label in google_updates
        ])

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

        model = genai.GenerativeModel('gemini-1.5-pro') # Or 'gemini-pro'
        response = model.generate_content(prompt)
        analysis = response.text.replace('•', '*')
        return analysis

    except genai.types.generation_types.BlockedPromptException:
         st.error("🔴 Gemini API Error: The prompt was blocked due to safety settings.")
         return "Analysis failed: Prompt was blocked by safety filters."
    except Exception as e:
        st.error(f"🔴 An error occurred while generating the AI deviation analysis: {e}")
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
            if 'Date' not in df.columns or 'Sessions' not in df.columns:
                st.error("🔴 Error: CSV must contain 'Date' and 'Sessions' columns.")
                return None

            try:
                date_str_series = df['Date'].astype(str)
                if not date_str_series.str.match(r'^\d{8}$').all():
                    raise ValueError("Some 'Date' values are not in YYYYMMDD format.")

                df['Sessions'] = pd.to_numeric(df['Sessions'], errors='coerce')
                if df['Sessions'].isnull().any():
                    st.warning("⚠️ Non-numeric 'Sessions' values found and ignored.")
                    df.dropna(subset=['Sessions'], inplace=True)

                if df.empty:
                     st.error("🔴 Error: No valid numeric 'Sessions' data found.")
                     return None

                df['Sessions'] = df['Sessions'].astype(int)
                df['Date'] = pd.to_datetime(date_str_series, format='%Y%m%d')
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
    return None

# --- Prophet Modeling and Plotting Function ---
def run_prophet_and_plot(df_original, effective_end_date, google_updates, granularity):
    """
    Fits Prophet, predicts historical fit & future forecast, plots results using Matplotlib.

    Args:
        df_original (pd.DataFrame): DF with 'Date' (datetime) and 'Sessions'.
        effective_end_date (pd.Timestamp): Furthest date for Prophet prediction.
        google_updates (list): List of Google update tuples.
        granularity (str): 'Daily', 'Weekly', or 'Monthly'.

    Returns:
        tuple: (full_forecast_df, historical_data_with_fit, last_actual_date)
               Returns (None, None, None) on failure.
    """
    df = df_original.copy()
    df.rename(columns={'Date': 'ds', 'Sessions': 'y'}, inplace=True)

    df_prophet = df
    resample_freq = 'D'
    if granularity == 'Weekly':
        resample_freq = 'W'
        try:
            df_prophet = df.set_index('ds').resample(resample_freq).sum().reset_index()
        except Exception as e:
             st.error(f"Error during weekly resampling: {e}")
             return None, None, None
    elif granularity == 'Monthly':
        resample_freq = 'M'
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

    full_forecast_df = None
    try:
        if effective_end_date < last_actual_date:
            st.warning(f"Effective end date ({effective_end_date.date()}) is before last actual date ({last_actual_date.date()}). Analyzing historical fit only.")
            effective_end_date = last_actual_date

        periods = 0
        if effective_end_date > last_actual_date:
             if granularity == 'Daily':
                 periods = (effective_end_date - last_actual_date).days
             elif granularity == 'Weekly':
                 periods = math.ceil((effective_end_date - last_actual_date).days / 7)
             elif granularity == 'Monthly':
                  temp_date = last_actual_date
                  while temp_date < effective_end_date:
                      temp_date += pd.offsets.MonthEnd(1)
                      if temp_date <= effective_end_date:
                          periods += 1

        model = Prophet()
        if granularity == 'Weekly':
            model.add_seasonality(name='yearly', period=365.25/7, fourier_order=5)
        if granularity == 'Monthly':
             model.add_seasonality(name='yearly', period=12, fourier_order=5, mode='multiplicative')

        model.fit(df_prophet)
        future_df = model.make_future_dataframe(periods=periods, freq=resample_freq, include_history=True)
        full_forecast_df = model.predict(future_df)

    except Exception as e:
        st.error(f"Error during Prophet modeling/prediction ({granularity}): {e}")
        st.error(traceback.format_exc())
        historical_data_with_fit_on_error = pd.merge(
             df_prophet, pd.DataFrame({'ds': df_prophet['ds']}), on='ds', how='left'
        ).assign(yhat=None, yhat_lower=None, yhat_upper=None)
        return None, historical_data_with_fit_on_error, last_actual_date

    historical_data_with_fit = pd.merge(
        df_prophet,
        full_forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='left'
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    try:
        ax.plot(historical_data_with_fit['ds'], historical_data_with_fit['y'],
                label=f'Actual ({granularity})', color='blue', marker='.', markersize=5, linestyle='-', zorder=3)
        ax.plot(historical_data_with_fit['ds'], historical_data_with_fit['yhat'],
                label='Prophet Predicted Fit (Historical)', color='orange', linestyle=':', linewidth=1.5, zorder=2)

        future_forecast_part = full_forecast_df[full_forecast_df['ds'] > last_actual_date]
        if not future_forecast_part.empty:
            ax.plot(future_forecast_part['ds'], future_forecast_part['yhat'],
                    label='Prophet Future Forecast', color='green', linestyle='--', linewidth=2, zorder=2)

        ax.fill_between(full_forecast_df['ds'], full_forecast_df['yhat_lower'], full_forecast_df['yhat_upper'],
                       color='skyblue', alpha=0.3, label='Confidence Interval (80%)', zorder=1)

        update_labels_added = set()
        for start_str, end_str, label in google_updates:
            try:
                start_date = pd.to_datetime(start_str, format='%Y%m%d')
                end_date = pd.to_datetime(end_str, format='%Y%m%d')
                mid_date = start_date + (end_date - start_date) / 2

                data_start_date = historical_data_with_fit['ds'].min()
                data_end_date = full_forecast_df['ds'].max()
                if start_date <= data_end_date and end_date >= data_start_date:
                    span_label = 'Google Update Period' if 'Google Update Period' not in update_labels_added else '_nolegend_'
                    ax.axvspan(start_date, end_date, color='lightcoral', alpha=0.25, label=span_label, zorder=0)
                    update_labels_added.add('Google Update Period')

                    y_limits = ax.get_ylim()
                    y_max_in_view = historical_data_with_fit['y'].max()
                    text_y_pos = (y_limits[1] * 1.02) if y_limits and y_limits[1] > y_limits[0] else (y_max_in_view * 1.05 if pd.notna(y_max_in_view) else 0)
                    ax.text(mid_date, text_y_pos, label, ha='center', va='bottom', fontsize=7, rotation=90, color='dimgray', zorder=4)

            except Exception as plot_err:
                 st.warning(f"Could not plot Google update '{label}': {plot_err}")

        ax.set_title(f'{granularity} Actual vs. Prophet Fit & Forecast with Google Updates')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sessions')
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during Matplotlib plotting ({granularity}): {e}")
    finally:
        plt.close(fig)

    return full_forecast_df, historical_data_with_fit, last_actual_date


# --- Future Forecast Dashboard Display ---
def display_dashboard(full_forecast_df, last_actual_date, forecast_end_date, granularity_label):
    """Displays table and summary for the FUTURE forecast part."""
    st.subheader(f"Future Forecast Summary ({granularity_label})")

    forecast_future = full_forecast_df[
        (full_forecast_df['ds'] > last_actual_date) &
        (full_forecast_df['ds'] <= forecast_end_date)
    ].copy()

    if not forecast_future.empty:
        st.write("Forecasted Values:")
        st.dataframe(forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].astype(
            {'yhat':'int', 'yhat_lower':'int', 'yhat_upper':'int'}
        ).assign(ds=lambda x: x['ds'].dt.strftime('%Y-%m-%d'))
          .reset_index(drop=True), height=300)

        horizon = len(forecast_future)
        unit_map = {'Daily': 'day', 'Weekly': 'week', 'Monthly': 'month'}
        unit = unit_map.get(granularity_label, 'period')
        horizon_str = f"{horizon} {unit}{'s' if horizon != 1 else ''}"

        st.write("Summary Metrics:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Last Actual Date", value=f"{last_actual_date.date() if last_actual_date else 'N/A'}")
        with col2:
            st.metric(label="Forecast Horizon Shown", value=f"{horizon_str}")

        forecast_value_at_end = forecast_future.iloc[-1]
        forecast_range = int(forecast_value_at_end['yhat_upper'] - forecast_value_at_end['yhat_lower'])
        delta_val = forecast_range / 2
        with col3:
            st.metric(label=f"Forecast at {forecast_value_at_end['ds'].strftime('%Y-%m-%d')}",
                      value=int(forecast_value_at_end['yhat']),
                      delta=f"±{delta_val:.0f} (Range: {forecast_range})",
                      delta_color="off")
    else:
        st.info("No future forecast points fall within the selected date range.")
        st.metric(label="Last Actual Date", value=f"{last_actual_date.date() if last_actual_date else 'N/A'}")

# --- HoloViews Animated Forecast Chart Function ---
def create_gapminder_forecast_chart(forecast_df, last_actual_date, granularity_label):
    """
    Creates an animated HoloViews chart visualizing the forecast over time,
    with point size representing the confidence interval range.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing Prophet forecast results
                                    ('ds', 'yhat', 'yhat_lower', 'yhat_upper').
        last_actual_date (pd.Timestamp): The last date of actual historical data.
        granularity_label (str): 'Daily', 'Weekly', or 'Monthly' for the title.

    Returns:
        hv.Points: The HoloViews chart object, or None if error.
    """
    try:
        # Filter for future dates only
        future_forecast = forecast_df[forecast_df['ds'] > last_actual_date].copy()

        if future_forecast.empty:
            st.info("ℹ️ No future forecast data points to animate.")
            return None

        # --- Prepare data for HoloViews ---
        future_forecast['ds'] = pd.to_datetime(future_forecast['ds']) # Ensure it's datetime
        future_forecast['uncertainty'] = (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).clip(lower=1)
        future_forecast['yhat_int'] = future_forecast['yhat'].round().astype(int)
        future_forecast['yhat_lower_int'] = future_forecast['yhat_lower'].round().astype(int)
        future_forecast['yhat_upper_int'] = future_forecast['yhat_upper'].round().astype(int)
        future_forecast['uncertainty_int'] = (future_forecast['yhat_upper_int'] - future_forecast['yhat_lower_int']).clip(lower=0)

        min_val = future_forecast['yhat_lower'].min()
        max_val = future_forecast['yhat_upper'].max()
        min_uncertainty = future_forecast['uncertainty'].min()
        max_uncertainty = future_forecast['uncertainty'].max()

        base_size = 5
        size_multiplier = 25
        if max_uncertainty <= min_uncertainty:
             future_forecast['size_scaled'] = base_size + size_multiplier / 2
        else:
            future_forecast['size_scaled'] = base_size + size_multiplier * (future_forecast['uncertainty'] - min_uncertainty) / (max_uncertainty - min_uncertainty)

        future_forecast['size_scaled'] = future_forecast['size_scaled'].fillna(base_size).round().astype(int)

        # --- Create HoloViews Points Element ---
        kdims = ['ds', 'yhat_int']
        vdims = ['yhat_lower_int', 'yhat_upper_int', 'uncertainty_int', 'size_scaled']
        points_plot = hv.Points(future_forecast, kdims=kdims, vdims=vdims, label="Forecast Point")

        # --- Define Tooltips ---
        # Formatting is handled within the tooltip string definition using Bokeh syntax
        tooltips = [
            ('Date', '@ds{%F}'),             # Format date using Bokeh field specifier
            ('Forecast (yhat)', '@yhat_int'),
            ('Lower CI', '@yhat_lower_int'),
            ('Upper CI', '@yhat_upper_int'),
            ('Uncertainty Range', '@uncertainty_int'),
            ('Scaled Size', '@size_scaled'),
        ]

        # --- Customize the plot using .opts ---
        animated_plot = points_plot.opts(
            hv.opts.Points(
                title=f"Animated {granularity_label} Forecast (Size = Uncertainty Range)",
                xlabel="Date",
                ylabel="Forecasted Sessions (yhat)",
                size='size_scaled',
                color='green',
                alpha=0.7,
                tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
                hover_tooltips=tooltips, # Use the correct option name
                # No separate 'formatters' needed here for hover tooltips
                ylim=(min_val * 0.95, max_val * 1.05) if pd.notna(min_val) and pd.notna(max_val) and min_val < max_val else None,
                width=800,
                height=450,
                # No 'xaxis' needed, type is inferred from data
                show_legend=False
             )
        )

        return animated_plot

    except ImportError:
         st.error("🔴 Error: HoloViews, Bokeh, or st_bokeh is not installed. Please install them (`pip install holoviews bokeh streamlit-bokeh`) to use the animated chart.")
         return None
    except Exception as e:
        st.error(f"🔴 An error occurred while creating the animated forecast chart: {e}")
        st.error(traceback.format_exc()) # Keep traceback for debugging
        return None

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Google Algorithm Impact Analyzer", layout="wide")
    st.title("📈 Google Algorithm Impact Analyzer")
    st.write("""
        Upload GA4 sessions CSV ('Date' as YYYYMMDD, 'Sessions').
        Visualize **Actual Traffic** vs. **Prophet's Predicted Fit** for the historical period, overlaid with Google updates.
        Optionally generate a future forecast, get AI analysis of **historical deviations**, and view an **animated forecast**.
    """)
    st.info("💡 Ensure CSV has 'Date' (YYYYMMDD format) and 'Sessions' (numeric) columns. You might need to `pip install holoviews bokeh streamlit-bokeh` for the animated chart.")

    # --- Define Google Updates List ---
    google_updates = [
        ('20230315', '20230328', 'Mar 2023 Core Update'), ('20230822', '20230907', 'Aug 2023 Core Update'),
        ('20230914', '20230928', 'Sept 2023 Helpful Content Update'), ('20231004', '20231019', 'Oct 2023 Spam Update'),
        ('20231005', '20231019', 'Oct 2023 Core Update'),
        ('20231102', '20231128', 'Nov 2023 Core Update'), ('20231108', '20231117', 'Nov 2023 Reviews Update'),
        ('20240305', '20240419', 'Mar 2024 Core Update'), ('20240305', '20240320', 'Mar 2024 Spam Update'),
        ('20240425', '20240425', 'Apr 2024 Reviews System Change'),
        ('20240505', '20240505', 'May 2024 Site Reputation Abuse'),
        ('20240506', '20240507', 'Site Rep Abuse'), ('20240514', '20240515', 'AI Overviews'),
        ('20240620', '20240627', 'June 2024 Core Update'),
        ('20240815', '20240903', 'Aug 2024 Core Update'),
    ]

    # --- Sidebar Controls ---
    granularity = st.sidebar.radio("Select Analysis Granularity", ("Daily", "Weekly", "Monthly"), key="granularity_radio")
    show_future_forecast = st.sidebar.checkbox("Include Future Forecast?", value=True, key="show_future_cb")

    user_selected_forecast_end_date = None

    if show_future_forecast:
        default_forecast_end = (pd.Timestamp.today() + timedelta(days=90)).date()
        min_date_allowed = pd.Timestamp.today().date() + timedelta(days=1)
        forecast_end_date_input = st.sidebar.date_input(
            "Select Forecast End Date",
            value=default_forecast_end,
            min_value=min_date_allowed,
            key="forecast_date_input"
        )
        user_selected_forecast_end_date = pd.to_datetime(forecast_end_date_input)
    else:
         st.sidebar.caption("Forecasting disabled.")


    # --- File Upload ---
    df_original = load_data()

    # --- Main Processing Area ---
    if df_original is not None:
        st.subheader("Data Preview (First 5 Rows)")
        st.dataframe(df_original.head().assign(Date=lambda x: x['Date'].dt.strftime('%Y-%m-%d')))

        # --- Run Prophet & Plotting ---
        st.markdown("---")
        st.header("📊 Historical Performance vs. Prophet Fit & Forecast")

        full_forecast_df = None
        historical_data_with_fit = None
        last_actual_date = None

        effective_end_date_for_prophet = user_selected_forecast_end_date if show_future_forecast and user_selected_forecast_end_date else df_original['Date'].max()

        try:
            with st.spinner(f"Running Prophet ({granularity}) and generating static plot..."):
                results = run_prophet_and_plot(
                    df_original.copy(),
                    effective_end_date_for_prophet,
                    google_updates,
                    granularity
                )

            if results:
                full_forecast_df, historical_data_with_fit, last_actual_date = results

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
             st.warning("⚠️ Cannot perform AI analysis: Historical data with Prophet fit not available (Prophet modeling might have failed).")
        else:
            if st.button("📈 Analyze Historical Performance Deviations", key="analyze_deviation_button"):
                if configure_gemini():
                    with st.spinner("🧠 Analyzing historical deviations with Gemini..."):
                        deviation_analysis_result = get_gemini_historical_deviation_analysis(
                            historical_data_with_fit.copy(),
                            google_updates
                        )
                    st.markdown(deviation_analysis_result)

        # --- Display Future Forecast Dashboard & Animated Chart Section ---
        st.markdown("---")
        if show_future_forecast:
            if full_forecast_df is not None and last_actual_date is not None and user_selected_forecast_end_date is not None:
                # 1. Display the summary dashboard FIRST
                display_dashboard(full_forecast_df, last_actual_date, user_selected_forecast_end_date, granularity)

                # 2. Add the Animated HoloViews Chart section
                st.subheader(f"🎬 Animated {granularity} Forecast Visualization")
                st.write("""
                    This chart shows the forecasted value (`yhat`) progressing over time.
                    The *size* of the point represents the width of the 80% confidence interval (uncertainty: `yhat_upper` - `yhat_lower`) for that point.
                    Hover over points for details. Use the slider/play button below the chart to navigate through time.
                """)
                with st.spinner("Generating animated forecast chart..."):
                    animated_chart = create_gapminder_forecast_chart(full_forecast_df, last_actual_date, granularity)

                if animated_chart:
                    try:
                        # Render the HoloViews object to a Bokeh figure/layout
                        bokeh_plot = hv.render(animated_chart, backend='bokeh')
                        # *** Use st_bokeh instead of st.bokeh_chart ***
                        st_bokeh(bokeh_plot, use_container_width=True)
                    except Exception as render_err:
                         st.error(f"🔴 Error rendering the animated chart: {render_err}")
                         st.error(traceback.format_exc()) # More detail on rendering error

            elif full_forecast_df is None or last_actual_date is None:
                 st.warning("Future forecast could not be generated (required for dashboard & animated chart). Check previous errors.")
            elif user_selected_forecast_end_date is None:
                 st.warning("Future forecast end date not properly set (required for dashboard & animated chart).") # Check if forecast checkbox is ticked but date somehow None
        else:
            st.caption("Future forecast display disabled via sidebar option (dashboard & animated chart also disabled).")


    else:
        uploader_state = st.session_state.get("ga4_csv_uploader")
        if uploader_state is None:
              st.info("Awaiting CSV file upload...")

    # --- Footer ---
    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")


if __name__ == "__main__":
    main()
