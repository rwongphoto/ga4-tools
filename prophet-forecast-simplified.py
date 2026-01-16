import os
import math
from datetime import timedelta

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import google.generativeai as genai
import traceback
import plotly.express as px
import streamlit.components.v1 as components

# --- Google Gemini Configuration ---
def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception:
        return False

# --- AI Deviation Analysis ---
def get_gemini_historical_deviation_analysis(historical_data_with_fit, google_updates):
    if not configure_gemini():
        return "Gemini API not configured. Analysis cannot proceed."
    if historical_data_with_fit is None or historical_data_with_fit.empty:
        return "Historical data with fit is empty. Cannot perform analysis."

    df = historical_data_with_fit.copy()
    df['deviation'] = df['y'] - df['yhat']
    df['deviation_pct'] = df.apply(
        lambda r: (r['deviation'] / r['yhat'] * 100) if r['yhat'] and abs(r['yhat']) > 1e-6 else 0,
        axis=1
    ).fillna(0).replace([float('inf'), -float('inf')], 0)

    # Prepare prompt summary
    start_date = df['ds'].min().strftime('%Y-%m-%d')
    end_date   = df['ds'].max().strftime('%Y-%m-%d')
    avg_dev    = df['deviation_pct'].mean()
    deviation_summary = (
        f"- Data Period: {start_date} to {end_date}\n"
        f"- Avg Deviation: {avg_dev:.2f}%\n"
    )
    updates_str = "\n".join(
        f"- {label} ({pd.to_datetime(s, format='%Y%m%d').strftime('%Y-%m-%d')} to "
        f"{pd.to_datetime(e, format='%Y%m%d').strftime('%Y-%m-%d')})"
        for s, e, label in google_updates
    )

    prompt = f"""
Analyze actual vs. predicted SEO traffic deviations:

{deviation_summary}

Google Updates:
{updates_str}

Provide 3–5 bullet points correlating positive/negative deviations with updates. Analysis should also consider trajectory traffic changes following the updates.
"""
    try:
        model    = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return response.text.replace('•', '*')
    except genai.types.generation_types.BlockedPromptException:
        return "Analysis failed: Prompt was blocked."
    except Exception as e:
        return f"Analysis error: {e}"

# --- CSV Loading & Validation ---
def load_data():
    uploaded_file = st.file_uploader("Choose a GA4 CSV file", type="csv", key="ga4_csv_uploader")
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
        if 'Date' not in df.columns or 'Sessions' not in df.columns:
            st.error("🔴 CSV must contain 'Date' and 'Sessions'.")
            return None
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')
        df['Sessions'] = pd.to_numeric(df['Sessions'], errors='coerce')
        df = df.dropna(subset=['Date','Sessions'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"🔴 Error loading CSV: {e}")
        return None

# --- Prophet Modeling & Static Plot ---
def run_prophet_and_plot(df_original, effective_end_date, google_updates, granularity):
    df = df_original.rename(columns={'Date':'ds','Sessions':'y'}).copy()
    # Resample
    if granularity == 'Weekly':
        df = df.set_index('ds').resample('W').sum().reset_index()
    elif granularity == 'Monthly':
        df = df.set_index('ds').resample('M').sum().reset_index()

    last_actual_date = df['ds'].max()
    # Calculate periods into future
    periods = 0
    if effective_end_date > last_actual_date:
        delta_days = (effective_end_date - last_actual_date).days
        if granularity == 'Daily':
            periods = delta_days
        elif granularity == 'Weekly':
            periods = math.ceil(delta_days / 7)
        elif granularity == 'Monthly':
            tmp = last_actual_date
            while tmp < effective_end_date:
                tmp += pd.offsets.MonthEnd(1)
                if tmp <= effective_end_date:
                    periods += 1

    m = Prophet()
    if granularity == 'Weekly':
        m.add_seasonality(name='yearly', period=365.25/7, fourier_order=5)
    if granularity == 'Monthly':
        m.add_seasonality(name='yearly', period=12, fourier_order=5, mode='multiplicative')
    m.fit(df)

    future = m.make_future_dataframe(periods=periods,
                                     freq={'Daily':'D','Weekly':'W','Monthly':'M'}[granularity],
                                     include_history=True)
    forecast = m.predict(future)

    hist_fit = pd.merge(
        df, forecast[['ds','yhat','yhat_lower','yhat_upper']],
        on='ds', how='left'
    )

    # Static Matplotlib plot
    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(hist_fit['ds'], hist_fit['y'], marker='.', label='Actual')
    ax.plot(hist_fit['ds'], hist_fit['yhat'], linestyle=':', label='Fit')
    future_part = forecast[forecast['ds'] > last_actual_date]
    if not future_part.empty:
        ax.plot(future_part['ds'], future_part['yhat'], linestyle='--', label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    alpha=0.3, label='CI (80%)')

    # Plot Google update spans
    first_update = True
    for s,e,label in google_updates:
        sd, ed = pd.to_datetime(s, format='%Y%m%d'), pd.to_datetime(e, format='%Y%m%d')
        if sd <= hist_fit['ds'].max() and ed >= hist_fit['ds'].min():
            span_label = 'Google Update' if first_update else '_nolegend_'
            ax.axvspan(sd, ed, color='lightcoral', alpha=0.2, label=span_label)
            first_update = False

    ax.set_title(f"{granularity} Actual vs. Prophet Fit & Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Sessions")
    ax.legend(loc='upper left'); ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    plt.close(fig)

    return forecast, hist_fit, last_actual_date

# --- Animated Forecast with Plotly Express ---
def plot_animated_forecast(hist_fit_df, full_forecast_df, google_updates):
    """
    Animated cumulative chart of actual vs. forecast,
    with Google algorithm update overlays.
    
    hist_fit_df   : DataFrame with columns ['ds','y','yhat'] for historical dates.
    full_forecast_df : DataFrame with ['ds','yhat'] covering history+future.
    """
    # 1) Build combined df: historical (y + yhat) + future (yhat only)
    last_hist = hist_fit_df['ds'].max()
    future_only = (
        full_forecast_df[full_forecast_df['ds'] > last_hist]
        [['ds','yhat']]
        .assign(y=pd.NA)
    )
    combined = pd.concat(
        [hist_fit_df[['ds','y','yhat']], future_only],
        ignore_index=True
    ).sort_values('ds')

    # 2) Build cumulative frames
    frames = []
    for T in combined['ds'].unique():
        sub = combined[combined['ds'] <= T].copy()
        sub['frame'] = T.strftime("%Y-%m-%d")
        frames.append(sub)
    anim_df = pd.concat(frames, ignore_index=True)

    # 3) Melt into long form
    anim_long = anim_df.melt(
        id_vars=['ds','frame'],
        value_vars=['y','yhat'],
        var_name='Type',
        value_name='Sessions'
    )
    anim_long['Type'] = anim_long['Type'].map({'y':'Actual','yhat':'Forecast'})

    # 4) Plotly Express animation
    fig = px.line(
        anim_long,
        x='ds',
        y='Sessions',
        color='Type',
        animation_frame='frame',
        animation_group='Type',
        labels={'ds':'Date','Sessions':'Sessions','Type':''},
        title='Actual vs. Forecast Animation'
    )
    fig.update_traces(mode='lines+markers')

    # 5) Lock axes
    y_min = anim_long['Sessions'].min(skipna=True)
    y_max = anim_long['Sessions'].max(skipna=True)
    fig.update_layout(
        xaxis=dict(range=[combined['ds'].min(), combined['ds'].max()]),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(title_text='')
    )

    # 6) Add Google‐update spans
    shapes = []
    for start_str, end_str, _ in google_updates:
        sd = pd.to_datetime(start_str, format='%Y%m%d')
        ed = pd.to_datetime(end_str,   format='%Y%m%d')
        shapes.append({
            'type':'rect','xref':'x','yref':'paper',
            'x0':sd,'x1':ed,'y0':0,'y1':1,
            'fillcolor':'LightCoral','opacity':0.2,'layer':'below','line_width':0,
            'name':'Google Update','showlegend':False
        })
    fig.update_layout(shapes=shapes)

    # 7) Render
    st.subheader("🎞️ Animated Actual vs. Forecast with Google Updates")
    st.plotly_chart(fig, use_container_width=True)



# --- Future Forecast Dashboard Display ---
def display_dashboard(full_forecast_df, last_actual_date, forecast_end_date, granularity_label):
    """
    Displays the FUTURE forecast table and summary metrics.
    Converts dates to strings so st.metric accepts them.
    """
    st.subheader(f"Future Forecast Summary ({granularity_label})")

    # Filter only the future portion of the forecast
    future_df = full_forecast_df[
        (full_forecast_df['ds'] > last_actual_date) &
        (full_forecast_df['ds'] <= forecast_end_date)
    ].copy()

    if not future_df.empty:
        st.write("Forecasted Values:")
        st.dataframe(
            future_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            .astype({'yhat':'int', 'yhat_lower':'int', 'yhat_upper':'int'})
            .rename(columns={'ds':'Date', 'yhat':'Forecast', 'yhat_lower':'Lower CI', 'yhat_upper':'Upper CI'})
            .reset_index(drop=True),
            height=300
        )

        # Summary metrics in three columns
        col1, col2, col3 = st.columns(3)
        with col1:
            # Convert date to string
            st.metric("Last Actual Date", last_actual_date.strftime("%Y-%m-%d"))
        with col2:
            horizon = len(future_df)
            unit = {'Daily':'day', 'Weekly':'week', 'Monthly':'month'}[granularity_label]
            st.metric("Forecast Horizon", f"{horizon} {unit}{'s' if horizon != 1 else ''}")
        with col3:
            last_row = future_df.iloc[-1]
            rng = last_row['yhat_upper'] - last_row['yhat_lower']
            st.metric(
                f"Forecast at {last_row['ds'].strftime('%Y-%m-%d')}",
                int(last_row['yhat']),
                delta=f"±{int(rng/2)} (Range: {int(rng)})"
            )
    else:
        st.info("No forecast points fall within the selected date range.")


# --- Main Application ---
def main():
    st.set_page_config(page_title="Google Algorithm Impact Analyzer", layout="wide")
    st.title("📈 Google Algorithm Impact Analyzer")
    st.write("""
        Upload GA4 CSV with 'Date' (YYYYMMDD) and 'Sessions'.
        View static & animated Prophet forecasts + AI-powered deviation analysis.
    """)

    # Google updates list
    google_updates = [
        ('20220825','20220909','Helpful Content Release'),
        ('20220912','20220926','Core Algorithm Update'),
        ('20220920','20220926','Product Review Algorithm Update'),
        ('20221019','20221021','Oct 2022 Spam Update'),         
        ('20221214','20230112','Dec 2022 Link Spam Update'),
        ('20230221','20230318','Feb 2023 Product Reviews Update'),        
        ('20230315','20230328','Mar 2023 Core Update'),
        ('20230822','20230907','Aug 2023 Core Update'),
        ('20230914','20230928','Sept 2023 Helpful Content Update'),
        ('20231004','20231019','Oct 2023 Core & Spam Updates'),
        ('20231102','20231204','Nov 2023 Core & Spam Updates'),
        ('20240305','20240419','Mar 2024 Core Update'),
        ('20240506','20240507','Site Rep Abuse'),
        ('20240514','20240515','AI Overviews'),
        ('20240620','20240627','June 2024 Core Update'),
        ('20240815','20240903','Aug 2024 Core Update'),
        ('20241111','20241205','Nov 2024 Core Update'),
        ('20241212','20241218','Dec 2024 Core Update'),
        ('20241219','20241226','Dec 2024 Spam Update'),
        ('20250313','20250327','Mar 2025 Core Update'),
        ('20250520','20250521','AI Mode US Launch'),
        ('20250630','20250717','Jun 2025 Core Update'),
        ('20250826','20250921','Aug 2025 Spam Update'),
        ('20251211','20251229','Dec 2025 Core Update')
    ]

    granularity = st.sidebar.radio("Select Analysis Granularity", ["Daily","Weekly","Monthly"])
    show_future_forecast = st.sidebar.checkbox("Include Future Forecast?", True)
    if show_future_forecast:
        default_end = pd.Timestamp.today() + timedelta(days=90)
        user_date = st.sidebar.date_input("Forecast End Date", default_end)
        forecast_end_date = pd.to_datetime(user_date)
    else:
        forecast_end_date = None

    df_original = load_data()
    if df_original is None:
        st.info("Awaiting CSV upload…")
        return

    st.subheader("Data Preview")
    st.dataframe(df_original.head().assign(Date=lambda d: d['Date'].dt.strftime('%Y-%m-%d')))

    st.markdown("---")
    with st.spinner(f"Running Prophet ({granularity})…"):
        end_date = forecast_end_date if show_future_forecast else df_original['Date'].max()
        full_forecast_df, hist_fit_df, last_actual_date = run_prophet_and_plot(
            df_original, end_date, google_updates, granularity
        )

    st.markdown("---")
    st.header("🤖 AI Analysis: Historical Deviations vs. Google Updates")
    if os.getenv("GOOGLE_API_KEY") and hist_fit_df is not None:
        if st.button("📈 Analyze Historical Deviations"):
            result = get_gemini_historical_deviation_analysis(hist_fit_df, google_updates)
            st.markdown(result)
    else:
        st.warning("AI analysis disabled (missing API key or data).")

    st.markdown("---")
    if full_forecast_df is not None:
        plot_animated_forecast(hist_fit_df, full_forecast_df, google_updates)

    st.markdown("---")
    if show_future_forecast and full_forecast_df is not None and last_actual_date is not None:
        display_dashboard(full_forecast_df, last_actual_date, forecast_end_date, granularity)

    st.markdown("---")
    st.markdown("Created by [The SEO Consultant.ai](https://theseoconsultant.ai/)")

if __name__ == "__main__":
    main()
