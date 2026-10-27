import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import json
import datetime
import os
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from fpdf import FPDF

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS3P8AUGWigUmZAVBBgxYYn1v8C3e-B3yPEPAGFeJSeisJ09aOwrZvadkkB5Ogo4LykEkEfK96vD18M/pub?output=csv"


def parse_float(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.endswith("%"):
        return float(s.strip("%")) / 100.0
    return float(s)


@st.cache_data(ttl=5)
def load_parameters(sheet_url):
    try:
        r = requests.get(sheet_url)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), header=None)

        params = {
            "start_date": pd.to_datetime(df.iloc[1, 1]).strftime("%Y-%m-%d"),
            "days_count": int(df.iloc[1, 2]),
            "distribution": str(df.iloc[4, 1]),
            "mean": parse_float(df.iloc[4, 2]),
            "deviation": parse_float(df.iloc[4, 3]),
            "correlation": parse_float(df.iloc[4, 4]),
            "dow_factors": [parse_float(x) for x in df.iloc[6, 1:8].tolist()],
            "growth": parse_float(df.iloc[7, 1]),
        }

        mines = []
        for col in range(1, df.shape[1]):
            v = df.iloc[2, col]
            if pd.notna(v) and str(v).strip():
                mines.append(str(v).strip())
        params["mines"] = mines

        events = []
        for row in range(9, 13):
            if row < len(df) and pd.notna(df.iloc[row, 1]):
                events.append({
                    "date": pd.to_datetime(df.iloc[row, 1]).strftime("%Y-%m-%d"),
                    "duration": int(df.iloc[row, 2]),
                    "factor": parse_float(df.iloc[row, 3]),
                    "prob": parse_float(df.iloc[row, 4]),
                })
        params["events"] = events
        return params
    except Exception as e:
        st.error(f"Failed to parse Google Sheet. Error: {e}")
        return None


@st.cache_data
def generate_simulation_data(params):
    dates = pd.date_range(start=params["start_date"], periods=params["days_count"])
    df = pd.DataFrame({"Date": dates})

    dow_factor = df["Date"].dt.dayofweek.apply(lambda x: params["dow_factors"][x]).to_numpy()
    trend = np.power(1 + params["growth"], np.arange(len(df)))

    event_factor = np.ones(len(df))
    for ev in params["events"]:
        ev_date = pd.to_datetime(ev["date"])
        mask = (df["Date"] >= ev_date) & (df["Date"] < ev_date + pd.Timedelta(days=ev["duration"]))
        for i in df[mask].index:
            day_in = (df.loc[i, "Date"] - ev_date).days
            bell = np.sin(np.pi * day_in / ev["duration"])
            if np.random.rand() <= ev["prob"]:
                event_factor[i] *= 1 + (ev["factor"] - 1) * bell

    for mine in params["mines"]:
        if params["distribution"].lower() == "normal":
            noise = np.random.normal(params["mean"], params["deviation"], len(df))
        else:
            lo = params["mean"] - params["deviation"]
            hi = params["mean"] + params["deviation"]
            noise = np.random.uniform(lo, hi, len(df))

        smoothed = np.empty(len(df))
        smoothed[0] = noise[0]
        c = params["correlation"]
        for i in range(1, len(df)):
            smoothed[i] = c * smoothed[i - 1] + (1 - c) * noise[i]

        df[mine] = np.maximum(0, smoothed * dow_factor * trend * event_factor)

    df["Total"] = df[params["mines"]].sum(axis=1)
    return df


def detect_anomalies(series, test_type, **kwargs):
    if test_type == "IQR":
        m = kwargs.get("multiplier", 1.5)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - m * iqr) | (series > q3 + m * iqr)
    if test_type == "Z-Score":
        t = kwargs.get("threshold", 3.0)
        z = np.abs(stats.zscore(series, nan_policy="omit"))
        return pd.Series(z > t, index=series.index)
    if test_type == "Moving Average":
        w = kwargs.get("window", 7)
        t = kwargs.get("threshold", 0.3)
        ma = series.rolling(window=w, min_periods=1).mean()
        return (np.abs(series - ma) / ma) > t
    if test_type == "Grubbs":
        t = kwargs.get("threshold", 3.0)
        z = np.abs(stats.zscore(series, nan_policy="omit"))
        return pd.Series(z > t, index=series.index)
    return pd.Series(False, index=series.index)


def compute_metrics(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return {
        "Mean": round(float(series.mean()), 2),
        "Std Dev": round(float(series.std()), 2),
        "Median": round(float(series.median()), 2),
        "IQR": round(float(q3 - q1), 2),
    }


def compute_trendline(series, degree):
    x = np.arange(len(series))
    return np.poly1d(np.polyfit(x, series.to_numpy(), degree))(x)


def parse_degree(trend_type):
    return int(trend_type.split("(")[1][0]) if "Poly" in trend_type else 1


def classify_anomalies(df, target, anomalies_mask, trendline):
    """Return (spikes, drops) split relative to the trendline."""
    anom = df[anomalies_mask]
    if anom.empty:
        return anom, anom
    values = df.loc[anom.index, target].to_numpy()
    above = values > trendline[anom.index.to_numpy()]
    return anom[above], anom[~above]


def build_dashboard_chart(df, params, chart_type, trend_type, test_method, kwargs):
    degree = parse_degree(trend_type)
    series = df["Total"]
    trend = compute_trendline(series, degree)
    anom = df[detect_anomalies(series, test_method, **kwargs)]

    fig = go.Figure()
    if chart_type == "Stacked Bar":
        for mine in params["mines"]:
            fig.add_trace(go.Bar(x=df["Date"], y=df[mine], name=mine))
        fig.update_layout(barmode="stack")
        fig.add_trace(go.Scatter(x=df["Date"], y=trend, mode="lines",
                                 name=f"{trend_type} Trend",
                                 line=dict(dash="dash", color="purple")))
        if not anom.empty:
            fig.add_trace(go.Scatter(x=anom["Date"], y=anom["Total"],
                                     mode="markers", name="Anomalies",
                                     marker=dict(color="red", size=12, symbol="x")))
    else:
        if chart_type == "Bar":
            fig.add_trace(go.Bar(x=df["Date"], y=series, name="Total"))
        else:
            fig.add_trace(go.Scatter(x=df["Date"], y=series, mode="lines", name="Total"))
        fig.add_trace(go.Scatter(x=df["Date"], y=trend, mode="lines",
                                 name=f"{trend_type} Trend",
                                 line=dict(dash="dash", color="purple")))
        if not anom.empty:
            fig.add_trace(go.Scatter(x=anom["Date"], y=anom["Total"],
                                     mode="markers", name="Anomalies",
                                     marker=dict(color="red", size=12, symbol="x")))

    fig.update_layout(
        title="Total Output Over Time",
        xaxis_title="Date", yaxis_title="Output",
        hovermode="x unified", template="plotly_white", height=500,
    )
    return fig


def build_pdf_chart_image(df, params, target, trend_type, test_method, kwargs):
    """One image per target: 3 subplots for Total (Line/Bar/Stacked), 2 for each mine."""
    is_total = target == "Total"
    n_rows = 3 if is_total else 2
    titles = ("Line + Trend", "Bar", "Stacked Bar") if is_total else ("Line + Trend", "Bar")

    degree = parse_degree(trend_type)
    series = df[target]
    trend = compute_trendline(series, degree)
    anom = df[detect_anomalies(series, test_method, **kwargs)]

    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles,
                        vertical_spacing=0.12)

    fig.add_trace(go.Scatter(x=df["Date"], y=series, mode="lines",
                             name=target, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=trend, mode="lines", name="Trend",
                             line=dict(dash="dash", color="purple"),
                             showlegend=False), row=1, col=1)
    if not anom.empty:
        fig.add_trace(go.Scatter(x=anom["Date"], y=anom[target], mode="markers",
                                 marker=dict(color="red", size=9, symbol="x"),
                                 showlegend=False), row=1, col=1)

    fig.add_trace(go.Bar(x=df["Date"], y=series, name=target,
                         showlegend=False), row=2, col=1)
    if not anom.empty:
        fig.add_trace(go.Scatter(x=anom["Date"], y=anom[target], mode="markers",
                                 marker=dict(color="red", size=9, symbol="x"),
                                 showlegend=False), row=2, col=1)

    if is_total:
        for mine in params["mines"]:
            fig.add_trace(go.Bar(x=df["Date"], y=df[mine], name=mine,
                                 showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=trend, mode="lines",
                                 line=dict(dash="dash", color="purple"),
                                 showlegend=False), row=3, col=1)

    fig.update_layout(
        height=350 * n_rows,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=30),
        barmode="stack" if is_total else "group",
    )
    fig.update_annotations(font_size=11)
    return fig.to_image(format="png", width=1000,
                        height=350 * n_rows, scale=1.5)


class PDFReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, "Weyland-Yutani Corporation", align="C")
        self.ln(5)
        self.set_font("helvetica", "I", 11)
        self.cell(0, 10, "Mining Ops Simulator - Detailed Analysis Report", align="C")
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _pdf_section(pdf, title, size=14):
    pdf.set_font("helvetica", "B", size)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(2)


def _pdf_metrics_table(pdf, metrics):
    pdf.set_font("helvetica", "B", 10)
    pdf.set_fill_color(220, 220, 220)
    for k in metrics:
        pdf.cell(45, 10, k, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_font("helvetica", "", 10)
    for v in metrics.values():
        pdf.cell(45, 10, str(v), border=1, align="C")
    pdf.ln(10)


def _pdf_anomaly_table(pdf, data, title, color, target, trendline):
    if data.empty:
        return
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("helvetica", "B", 9)
    pdf.cell(40, 8, "Date", border=1)
    pdf.cell(50, 8, "Value", border=1)
    pdf.cell(50, 8, "Deviation from Trend (%)", border=1)
    pdf.ln()

    pdf.set_font("helvetica", "", 9)
    for idx, row in data.iterrows():
        baseline = trendline[idx]
        dev_pct = ((row[target] - baseline) / baseline * 100) if baseline > 0 else 0.0
        pdf.cell(40, 8, row["Date"].strftime("%Y-%m-%d"), border=1)
        pdf.cell(50, 8, f"{row[target]:.2f}", border=1)
        pdf.cell(50, 8, f"{dev_pct:+.2f}%", border=1)
        pdf.ln()
    pdf.ln(5)


def _pdf_add_image(pdf, img_bytes, name):
    safe = re.sub(r"[^a-zA-Z0-9]", "_", name)
    tmp = f"temp_chart_{safe}.png"
    with open(tmp, "wb") as f:
        f.write(img_bytes)
    pdf.image(tmp, x=15, w=180)
    os.remove(tmp)


def generate_pdf(df, params, test_method, kwargs, trend_type):
    pdf = PDFReport()
    degree = parse_degree(trend_type)

    for target in ["Total"] + params["mines"]:
        pdf.add_page()
        series = df[target]
        trendline = compute_trendline(series, degree)
        anom_mask = detect_anomalies(series, test_method, **kwargs)
        spikes, drops = classify_anomalies(df, target, anom_mask, trendline)

        _pdf_section(pdf, f"Analysis for: {target}")
        _pdf_metrics_table(pdf, compute_metrics(series))

        _pdf_section(pdf, f"Detected Anomalies ({test_method})", size=12)
        if spikes.empty and drops.empty:
            pdf.set_font("helvetica", "I", 10)
            pdf.cell(0, 10, "No anomalies detected with the current parameters.", ln=True)
        else:
            _pdf_anomaly_table(pdf, spikes, "Spikes (Above Trend)", (200, 0, 0),
                               target, trendline)
            _pdf_anomaly_table(pdf, drops, "Drops (Below Trend)", (0, 0, 200),
                               target, trendline)
        pdf.ln(5)

        _pdf_section(pdf, "Production Charts", size=12)
        img = build_pdf_chart_image(df, params, target, trend_type, test_method, kwargs)
        _pdf_add_image(pdf, img, target)
        pdf.ln(5)

    return bytes(pdf.output())


@st.cache_data(show_spinner=False)
def _generate_pdf_cached(df_json, params_json, test_method, kwargs_json, trend_type):
    df_ = pd.read_json(io.StringIO(df_json))
    df_["Date"] = pd.to_datetime(df_["Date"])
    return generate_pdf(
        df_, json.loads(params_json), test_method,
        json.loads(kwargs_json), trend_type,
    )


st.set_page_config(page_title="Weyland-Yutani Mining Ops", layout="wide")
st.title("Weyland-Yutani Corp. | Mining Ops Simulator")

st.sidebar.header("Configuration")
st.sidebar.markdown("**To update data:** refresh the page. If the data didn't change, refresh again.")

params = load_parameters(SHEET_URL)
if not params:
    st.stop()

df = generate_simulation_data(params)

st.sidebar.header("Anomaly Detection Settings")
test_method = st.sidebar.selectbox(
    "Select Test", ["IQR", "Z-Score", "Moving Average", "Grubbs"]
)

kwargs = {}
if test_method == "IQR":
    kwargs["multiplier"] = st.sidebar.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
elif test_method == "Z-Score":
    kwargs["threshold"] = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
elif test_method == "Moving Average":
    kwargs["window"] = st.sidebar.slider("MA Window (Days)", 3, 14, 7)
    kwargs["threshold"] = st.sidebar.slider("Distance Threshold (%)", 0.05, 0.50, 0.20, 0.05)
elif test_method == "Grubbs":
    kwargs["threshold"] = st.sidebar.slider("Grubbs Critical Value", 2.0, 4.0, 3.0, 0.1)

st.sidebar.header("Chart Settings")
chart_type = st.sidebar.radio("Chart Type", ["Line", "Bar", "Stacked Bar"])
trend_type = st.sidebar.selectbox(
    "Trendline", ["Linear (1)", "Poly (2)", "Poly (3)", "Poly (4)"]
)

st.subheader("Statistical Summary (All Targets)")
rows = []
for t in params["mines"] + ["Total"]:
    row = {"Target": t}
    row.update(compute_metrics(df[t]))
    rows.append(row)
st.dataframe(pd.DataFrame(rows).set_index("Target"), use_container_width=True)

st.subheader("Production Trends & Anomalies (Total)")
st.plotly_chart(
    build_dashboard_chart(df, params, chart_type, trend_type, test_method, kwargs),
    use_container_width=True,
)

st.subheader("Report Generation")
st.markdown("Generate a detailed PDF containing metrics, charts, and anomaly logs for **all** mines and the Total.")

if st.button("Generate & Download PDF Report"):
    with st.spinner("Generating PDF... (first time ~8s, subsequent clicks are cached)"):
        pdf_bytes = _generate_pdf_cached(
            df.to_json(date_format="iso"),
            json.dumps(params, sort_keys=True, default=str),
            test_method,
            json.dumps(kwargs, sort_keys=True),
            trend_type,
        )
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name=f"WY_Mining_Report_{datetime.datetime.now():%Y%m%d}.pdf",
        mime="application/pdf",
    )
    st.success("PDF generated successfully!")