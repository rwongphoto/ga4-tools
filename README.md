# GA4 Tools

Streamlit forecasting tools for GA4 session data. Compares actual post-update traffic against a counterfactual forecast to quantify the impact of Google algorithm updates and support planning.

## Tools in this repo

- **`google-traffic-forecast.py`** — upload a GA4 CSV (`Date`, `Sessions`), fit a NeuralProphet model on pre-change history, and forecast forward. Use for baseline traffic projections and budget / headcount justification.
- **`google-updates-forecast.py`** — counterfactual impact analysis for Google core / spam / helpful-content updates. Trains on pre-update data, forecasts what traffic *would have been*, and overlays the actual post-update curve to show lift or loss.
- **`prophet-forecast-simplified.py`** — lighter-weight Prophet (not NeuralProphet) variant for quick forecasts when you don't need the NN components.

## Stack

- Streamlit UI
- NeuralProphet + Prophet for time-series forecasting
- Pandas for data shaping
- Matplotlib for plots

## Setup

```bash
pip install -r requirements.txt
streamlit run google-traffic-forecast.py
# or
streamlit run google-updates-forecast.py
# or
streamlit run prophet-forecast-simplified.py
```

## Input format

CSV with at least `Date` and `Sessions` columns. Daily granularity recommended; weekly works but reduces seasonality resolution.

## PyTorch note

NeuralProphet pins older PyTorch versions. If you hit a `safe_globals` / serialization error on PyTorch ≥ 2.6, pin to PyTorch < 2.6 or enable the allowlist block at the top of `google-traffic-forecast.py`.
