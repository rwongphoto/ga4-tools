import streamlit as st
import pandas as pd
import holoviews as hv
import panel as pn

hv.extension('bokeh')
pn.extension()

# Dummy data
df = pd.DataFrame({
    'ds': pd.date_range('2025-01-01', periods=12, freq='M'),
    'yhat': [50, 55, 60, 58, 62, 65, 70, 68, 72, 75, 80, 85],
    'yhat_lower': [45,50,55,53,58,60,65,63,68,70,75,80],
    'yhat_upper': [55,60,65,63,66,70,75,73,76,80,85,90],
})
last_actual = df['ds'].max() - pd.DateOffset(months=3)

# Just show the “future” points as circles sized by uncertainty
df_future = df[df['ds'] > last_actual].copy()
df_future['uncertainty'] = df_future['yhat_upper'] - df_future['yhat_lower']
df_future['size'] = (df_future['uncertainty'] / df_future['uncertainty'].max()) * 20 + 5

points = hv.Points(
    df_future,
    kdims=['ds', 'yhat'],
    vdims=['yhat_lower','yhat_upper','uncertainty','size']
).opts(
    title="Test Forecast",
    size='size',
    tools=['hover','wheel_zoom','pan'],
    width=800, height=400
)

# Render and display
bokeh_fig = hv.render(points, backend='bokeh')
st.bokeh_chart(bokeh_fig, use_container_width=True)

