import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import datetime
import plotly.graph_objects as go
from scipy import stats
from fpdf import FPDF
import os
import re
import plotly.io as pio
pio.kaleido.scope.chromium_args += ("--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage")

def parse_float(val):
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    val_str = str(val).strip()
    if val_str.endswith('%'): return float(val_str.strip('%')) / 100.0
    return float(val_str)

@st.cache_data(ttl=5)
def load_parameters(sheet_url):
    try:
        response = requests.get(sheet_url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), header=None)
        
        params = {}
        params['start_date'] = pd.to_datetime(df.iloc[1, 1]).strftime('%Y-%m-%d') 
        params['days_count'] = int(df.iloc[1, 2]) 
        
        mines = []
        for col in range(1, df.shape[1]):
            val = df.iloc[2, col]
            if pd.notna(val) and str(val).strip() != "":
                mines.append(str(val).strip())
        params['mines'] = mines
        
        params['distribution'] = str(df.iloc[4, 1]) 
        params['mean'] = parse_float(df.iloc[4, 2]) 
        params['deviation'] = parse_float(df.iloc[4, 3]) 
        params['correlation'] = parse_float(df.iloc[4, 4]) 
        
        params['dow_factors'] = [parse_float(x) for x in df.iloc[6, 1:8].tolist()] 
        params['growth'] = parse_float(df.iloc[7, 1]) 
        
        events = []
        for row in range(9, 13): 
            if row < len(df) and pd.notna(df.iloc[row, 1]):
                events.append({
                    'date': pd.to_datetime(df.iloc[row, 1]).strftime('%Y-%m-%d'),
                    'duration': int(df.iloc[row, 2]), 
                    'factor': parse_float(df.iloc[row, 3]), 
                    'prob': parse_float(df.iloc[row, 4])    
                })
        params['events'] = events
        return params
    except Exception as e:
        st.error(f"Failed to parse Google Sheet. Error: {e}")
        return None

@st.cache_data
def generate_simulation_data(params):
    
    dates = pd.date_range(start=params['start_date'], periods=params['days_count'])
    df = pd.DataFrame({'Date': dates})
    
    df['DOW'] = df['Date'].dt.dayofweek
    df['DOW_Factor'] = df['DOW'].apply(lambda x: params['dow_factors'][x])
    df['Trend'] = [(1 + params['growth']) ** i for i in range(len(df))]
    
    event_factor = np.ones(len(df))
    for ev in params['events']:
        ev_date = pd.to_datetime(ev['date'])
        mask = (df['Date'] >= ev_date) & (df['Date'] < ev_date + pd.Timedelta(days=ev['duration']))
        for i in df[mask].index:
            day_in_event = (df.loc[i, 'Date'] - ev_date).days
            bell = np.sin(np.pi * day_in_event / ev['duration'])
            trigger = np.random.rand() <= ev['prob']
            if trigger:
                event_factor[i] *= (1 + (ev['factor'] - 1) * bell)
    
    for mine in params['mines']:
        noise = []
        for _ in range(len(df)):
            if params['distribution'].lower() == 'normal':
                val = np.random.normal(params['mean'], params['deviation'])
            else:
                val = np.random.uniform(params['mean'] - params['deviation'], params['mean'] + params['deviation'])
            noise.append(val)
            
        smoothed = [noise[0]]
        for i in range(1, len(noise)):
            sm = (params['correlation'] * smoothed[-1]) + ((1 - params['correlation']) * noise[i])
            smoothed.append(sm)
            
        output = np.maximum(0, np.array(smoothed) * df['DOW_Factor'] * df['Trend'] * event_factor)
        df[mine] = output

    df['Total'] = df[params['mines']].sum(axis=1)
    return df.drop(columns=['DOW', 'DOW_Factor', 'Trend'])

def detect_anomalies(series, test_type, **kwargs):
    anomalies = pd.Series(False, index=series.index)
    if test_type == "IQR":
        multiplier = kwargs.get('multiplier', 1.5)
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        anomalies = (series < Q1 - multiplier * IQR) | (series > Q3 + multiplier * IQR)
    elif test_type == "Z-Score":
        threshold = kwargs.get('threshold', 3.0)
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        anomalies = z_scores > threshold
    elif test_type == "Moving Average":
        window = kwargs.get('window', 7)
        threshold = kwargs.get('threshold', 0.3)
        ma = series.rolling(window=window, min_periods=1).mean()
        distance = np.abs(series - ma) / ma
        anomalies = distance > threshold
    elif test_type == "Grubbs":
        threshold = kwargs.get('threshold', 3.0)
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        anomalies = z_scores > threshold
    return anomalies

def build_chart_image(df, target, chart_type, trend_type, test_method, kwargs):
    fig = go.Figure()
    
    if chart_type == "Stacked Bar" and target == 'Total':
        for mine in [c for c in df.columns if c not in ['Date', 'Total']]:
            fig.add_trace(go.Bar(x=df['Date'], y=df[mine], name=mine))
        fig.update_layout(barmode='stack')
        
        degree = int(trend_type.split('(')[1][0]) if "Poly" in trend_type else 1
        x_idx = np.arange(len(df))
        z = np.polyfit(x_idx, df[target], degree)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=df['Date'], y=p(x_idx), mode='lines', name='Trend', line=dict(dash='dash', color='purple')))
    else:
        if chart_type == "Bar":
            fig.add_trace(go.Bar(x=df['Date'], y=df[target], name=target))
        else:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[target], mode='lines', name=target))
            
        degree = int(trend_type.split('(')[1][0]) if "Poly" in trend_type else 1
        x_idx = np.arange(len(df))
        z = np.polyfit(x_idx, df[target], degree)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=df['Date'], y=p(x_idx), mode='lines', name='Trend', line=dict(dash='dash', color='purple')))
    
    anomalies_mask = detect_anomalies(df[target], test_method, **kwargs)
    anom_data = df[anomalies_mask]
    if not anom_data.empty:
        fig.add_trace(go.Scatter(
            x=anom_data['Date'], y=anom_data[target], 
            mode='markers', name='Anomalies',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
    fig.update_layout(
        title=f"{target} Output",
        xaxis_title="Date", yaxis_title="Output",
        template="plotly_white", height=400,
        showlegend=False
    )
    return fig.to_image(format="png", width=1000, height=400, scale=2)

class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, 'Weyland-Yutani Corporation', border=False, align='C')
        self.ln(5)
        self.set_font('helvetica', 'I', 11)
        self.cell(0, 10, 'Mining Ops Simulator - Detailed Analysis Report', border=False, align='C')
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(df, params, test_method, kwargs, trend_type):
    pdf = PDFReport()
    targets = ['Total'] + params['mines']
    
    degree = int(trend_type.split('(')[1][0]) if "Poly" in trend_type else 1
    x_idx = np.arange(len(df))
    
    for target in targets:
        pdf.add_page()
        
        pdf.set_font('helvetica', 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Analysis for: {target}", ln=True)
        pdf.ln(2)
        
        series = df[target]
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        metrics = {
            "Mean": round(series.mean(), 2),
            "Std Dev": round(series.std(), 2),
            "Median": round(series.median(), 2),
            "IQR": round(Q3 - Q1, 2)
        }
        
        pdf.set_font('helvetica', 'B', 10)
        pdf.set_fill_color(220, 220, 220)
        for k in metrics.keys():
            pdf.cell(45, 10, k, border=1, align='C', fill=True)
        pdf.ln()
        
        pdf.set_font('helvetica', '', 10)
        for v in metrics.values():
            pdf.cell(45, 10, str(v), border=1, align='C')
        pdf.ln(10)
        
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, f"Detected Anomalies ({test_method})", ln=True)
        pdf.ln(2)
        
        anomalies_mask = detect_anomalies(df[target], test_method, **kwargs)
        anom_data = df[anomalies_mask]
        
        z = np.polyfit(x_idx, series, degree)
        p = np.poly1d(z)
        trendline = p(x_idx) 
        
        if anom_data.empty:
            pdf.set_font('helvetica', 'I', 10)
            pdf.cell(0, 10, "No anomalies detected with the current parameters.", ln=True)
        else:
            spike_indices = [i for i in anom_data.index if series.loc[i] > trendline[i]]
            drop_indices = [i for i in anom_data.index if series.loc[i] <= trendline[i]]
            
            spikes = anom_data.loc[spike_indices]
            drops = anom_data.loc[drop_indices]
            
            def render_anomaly_table(data, title, color):
                if data.empty: return
                pdf.set_font('helvetica', 'B', 11)
                pdf.set_text_color(*color)
                pdf.cell(0, 8, title, ln=True)
                pdf.set_text_color(0, 0, 0)
                
                pdf.set_font('helvetica', 'B', 9)
                pdf.cell(40, 8, "Date", border=1)
                pdf.cell(50, 8, "Value", border=1)
                pdf.cell(50, 8, "Deviation from Trend (%)", border=1)
                pdf.ln()
                
                pdf.set_font('helvetica', '', 9)
                for idx, row in data.iterrows():
                    baseline = trendline[idx]
                    if baseline <= 0:
                        dev_pct = 0.0
                    else:
                        dev_pct = ((row[target] - baseline) / baseline) * 100
                    pdf.cell(40, 8, row['Date'].strftime('%Y-%m-%d'), border=1)
                    pdf.cell(50, 8, f"{row[target]:.2f}", border=1)
                    pdf.cell(50, 8, f"{dev_pct:+.2f}%", border=1)
                    pdf.ln()
                pdf.ln(5)
                
            render_anomaly_table(spikes, "Spikes (Above Trend)", (200, 0, 0))
            render_anomaly_table(drops, "Drops (Below Trend)", (0, 0, 200))
            
        pdf.ln(5)
        
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 10, "Production Charts", ln=True)
        pdf.ln(5)
        
        safe_target = re.sub(r'[^a-zA-Z0-9]', '_', target)
        
        img_line = build_chart_image(df, target, "Line", trend_type, test_method, kwargs)
        temp_img = f"temp_line_{safe_target}.png"
        with open(temp_img, "wb") as f: f.write(img_line)
        pdf.image(temp_img, x=15, w=180)
        os.remove(temp_img)
        pdf.ln(5)
        
        img_bar = build_chart_image(df, target, "Bar", trend_type, test_method, kwargs)
        temp_img = f"temp_bar_{safe_target}.png"
        with open(temp_img, "wb") as f: f.write(img_bar)
        pdf.image(temp_img, x=15, w=180)
        os.remove(temp_img)
        pdf.ln(5)
        
        if target == 'Total':
            img_stack = build_chart_image(df, target, "Stacked Bar", trend_type, test_method, kwargs)
            temp_img = f"temp_stack_{safe_target}.png"
            with open(temp_img, "wb") as f: f.write(img_stack)
            pdf.image(temp_img, x=15, w=180)
            os.remove(temp_img)
            pdf.ln(5)

    pdf.output("Weyland_Yutani_Report.pdf")

st.set_page_config(page_title="Weyland-Yutani Mining Ops", layout="wide")
st.title("Weyland-Yutani Corp. | Mining Ops Simulator")

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

st.sidebar.header("Configuration")
sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vS3P8AUGWigUmZAVBBgxYYn1v8C3e-B3yPEPAGFeJSeisJ09aOwrZvadkkB5Ogo4LykEkEfK96vD18M/pub?output=csv"

st.sidebar.markdown("**To update data:** please refresh the page. If data didn't change - refresh again.")

params = load_parameters(sheet_url)
if params:
    df = generate_simulation_data(params)
    
    k_suffix = f"_{st.session_state.reset_counter}"
    
    st.sidebar.header("Anomaly Detection Settings")
    test_method = st.sidebar.selectbox(
        "Select Test", 
        ["IQR", "Z-Score", "Moving Average", "Grubbs"],
        key=f"test_method{k_suffix}"
    )
    
    kwargs = {}
    if test_method == "IQR":
        kwargs['multiplier'] = st.sidebar.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1, key=f"iqr_mult{k_suffix}")
    elif test_method == "Z-Score":
        kwargs['threshold'] = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key=f"z_thresh{k_suffix}")
    elif test_method == "Moving Average":
        kwargs['window'] = st.sidebar.slider("MA Window (Days)", 3, 14, 7, key=f"ma_window{k_suffix}")
        kwargs['threshold'] = st.sidebar.slider("Distance Threshold (%)", 0.05, 0.50, 0.20, 0.05, key=f"ma_thresh{k_suffix}")
    elif test_method == "Grubbs":
        kwargs['threshold'] = st.sidebar.slider("Grubbs Critical Value", 2.0, 4.0, 3.0, 0.1, key=f"grubbs_thresh{k_suffix}")

    st.sidebar.header("Chart Settings")
    chart_type = st.sidebar.radio("Chart Type", ["Line", "Bar", "Stacked Bar"], key=f"chart_type{k_suffix}")
    trend_type = st.sidebar.selectbox("Trendline", ["Linear (1)", "Poly (2)", "Poly (3)", "Poly (4)"], key=f"trend_type{k_suffix}")
    
    st.subheader("Statistical Summary (All Targets)")
    metrics_list = []
    for target in params['mines'] + ['Total']:
        s = df[target]
        metrics_list.append({
            "Target": target,
            "Mean": round(s.mean(), 2),
            "Std Dev": round(s.std(), 2),
            "Median": round(s.median(), 2),
            "IQR": round(s.quantile(0.75) - s.quantile(0.25), 2)
        })
    st.dataframe(pd.DataFrame(metrics_list).set_index("Target"), use_container_width=True)
    
    st.subheader("Production Trends & Anomalies (Total)")
    
    fig = go.Figure()
    
    if chart_type == "Stacked Bar":
        for mine in params['mines']:
            fig.add_trace(go.Bar(x=df['Date'], y=df[mine], name=mine))
        fig.update_layout(barmode='stack')
        
        degree = int(trend_type.split('(')[1][0]) if "Poly" in trend_type else 1
        x_idx = np.arange(len(df))
        z = np.polyfit(x_idx, df['Total'], degree)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=df['Date'], y=p(x_idx), mode='lines', name=f'{trend_type} Trend', line=dict(dash='dash', color='purple')))
    else:
        if chart_type == "Bar":
            fig.add_trace(go.Bar(x=df['Date'], y=df['Total'], name='Total'))
        else:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Total'], mode='lines', name='Total'))
            
        degree = int(trend_type.split('(')[1][0]) if "Poly" in trend_type else 1
        x_idx = np.arange(len(df))
        z = np.polyfit(x_idx, df['Total'], degree)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=df['Date'], y=p(x_idx), mode='lines', name=f'{trend_type} Trend', line=dict(dash='dash', color='purple')))

    anomalies_mask = detect_anomalies(df['Total'], test_method, **kwargs)
    anom_data = df[anomalies_mask]
    
    if not anom_data.empty:
        fig.add_trace(go.Scatter(
            x=anom_data['Date'], 
            y=anom_data['Total'], 
            mode='markers', 
            name='Anomalies',
            marker=dict(color='red', size=12, symbol='x')
        ))

    fig.update_layout(
        title="Total Output Over Time",
        xaxis_title="Date",
        yaxis_title="Output",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Report Generation")
    st.markdown("Choose the test and trendline to generate a detailed PDF containing metrics, charts, and anomaly logs for **all** mines and the total.")
    
    if st.button("Generate & Download PDF Report"):
        with st.spinner("Generating comprehensive PDF... (this takes a moment)"):
            generate_pdf(df, params, test_method, kwargs, trend_type)
            
            with open("Weyland_Yutani_Report.pdf", "rb") as pdf_file:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_file,
                    file_name=f"WY_Mining_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            st.success("PDF Generated Successfully!")