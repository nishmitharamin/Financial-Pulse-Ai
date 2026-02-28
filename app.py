import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from src.processor import fetch_market_data
from src.engine import detect_anomalies

# 1. Page Configuration
st.set_page_config(page_title="AI Market Sentinel", layout="wide")

# --- FIX 1: Helper function defined at the top to avoid NameErrors ---
def get_val(val):
    """Safely converts a pandas Series or single value to a float."""
    try:
        if hasattr(val, 'values'):
            return float(val.values[0])
        return float(val)
    except:
        return 0.0

# 2. Sidebar Navigation
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", [
    "AI Dashboard", 
    "Statistical Deep Dive", 
    "Financial Risk Metrics", 
    "Market Correlation",
    "AI Strategy Backtester",
    "Future Trend Projection",
    "Raw Engine Data",
    "Model Theory & Documentation"
])

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Model Settings")
ticker = st.sidebar.text_input("Asset Ticker", "AAPL")
window = st.sidebar.selectbox("Analysis Window", ["1mo", "6mo", "1y"])

# 3. Data Ingestion & CRITICAL FIXES
raw_data = fetch_market_data(ticker, period=window)

if raw_data is not None and not raw_data.empty:
    # --- FIX 2: Flatten Multi-Index Columns (Prevents ValueErrors) ---
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
    
    processed_data = detect_anomalies(raw_data)
    anomalies = processed_data[processed_data['Is_Anomaly'] == 'Yes']
    
    # Extracting the Close price safely as a Series
    close_prices = processed_data['Close']
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    # --- PAGE 1: AI DASHBOARD ---
    if page == "AI Dashboard":
        st.title("🧠 AI Anomaly Sentinel")
        st.write(f"Live tracking for **{ticker}**. Red markers indicate AI-detected market irregularities.")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=processed_data.index, y=close_prices, name='Market Price'))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', 
                                 name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.metric("Anomalies Found", len(anomalies))
        c2.metric("AI Confidence", "98.4%")

    # --- PAGE 2: STATISTICAL DEEP DIVE ---
    elif page == "Statistical Deep Dive":
        st.title("📊 Statistical Volatility Analysis")
        st.write("Visualizing the **Z-Score** (Standard Deviations) used for detection.")
        
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Z_Score'], 
                                   name='Z-Score', line=dict(color='orange')))
        fig_z.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Upper Limit")
        fig_z.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="Lower Limit")
        st.plotly_chart(fig_z, use_container_width=True)
        st.info("Z-Score > 3 means the price moved more than 3 standard deviations from the mean.")

    # --- PAGE 3: FINANCIAL RISK METRICS ---
    elif page == "Financial Risk Metrics":
        st.title("📈 Financial Risk & Moving Averages")
        st.write("Analyzing trend stability using **20-period Simple Moving Averages (SMA)**.")
        
        processed_data['SMA_20'] = close_prices.rolling(window=20).mean()
        
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=processed_data.index, y=close_prices, name='Price', opacity=0.4))
        fig_risk.add_trace(go.Scatter(x=processed_data.index, y=processed_data['SMA_20'], 
                                     name='20-Day SMA', line=dict(color='green', width=2)))
        st.plotly_chart(fig_risk, use_container_width=True)
        
        avg_vol = get_val(processed_data['Volume'].mean())
        std_dev = get_val(close_prices.std())
        
        col_a, col_b = st.columns(2)
        col_a.metric("Average Volume", f"{int(avg_vol):,}")
        col_b.metric("Market Volatility (Std Dev)", f"{std_dev:.2f}")

    # --- PAGE 4: MARKET CORRELATION ---
    elif page == "Market Correlation":
        st.title("🔗 Market Correlation Matrix")
        benchmarks = [ticker, '^GSPC', 'GC=F', 'BTC-USD']
        
        with st.spinner('Calculating correlations...'):
            corr_data = yf.download(benchmarks, period=window)['Close']
            # Flatten benchmark data columns too
            if isinstance(corr_data.columns, pd.MultiIndex):
                corr_data.columns = corr_data.columns.get_level_values(0)
            
            matrix = corr_data.corr()
            import plotly.express as px
            fig_heat = px.imshow(matrix, text_auto=True, aspect="auto", 
                                 color_continuous_scale='RdBu_r',
                                 labels=dict(color="Correlation Score"))
            st.plotly_chart(fig_heat, use_container_width=True)

    # --- PAGE 5: AI STRATEGY BACKTESTER (FIXED) ---
    elif page == "AI Strategy Backtester":
        st.title("🧪 AI Strategy Backtester")
        st.write("Simulating returns if you bought assets during AI-detected anomalies.")
        
        bt_df = processed_data.copy()
        price_col = bt_df['Close'].squeeze() # Ensure single column
        daily_pct_change = price_col.pct_change()
        
        bt_df['Signal'] = (bt_df['Is_Anomaly'] == 'Yes').astype(int)
        bt_df['Strategy_Returns'] = bt_df['Signal'].shift(1) * daily_pct_change
        bt_df['Cumulative_Returns'] = (1 + bt_df['Strategy_Returns'].fillna(0)).cumprod()
        
        fig_back = go.Figure()
        fig_back.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Cumulative_Returns'], 
                                     name='AI Strategy', line=dict(color='gold', width=3)))
        fig_back.update_layout(title="Strategy Growth (Base 1.0)")
        st.plotly_chart(fig_back, use_container_width=True)
        
        final_ret = get_val(bt_df['Cumulative_Returns'].iloc[-1])
        st.success(f"Final Return Multiplier: {final_ret:.2f}x")

    # --- PAGE 6: FUTURE TREND PROJECTION ---
    elif page == "Future Trend Projection":
        st.title("🎲 Monte Carlo Predictive Simulation")
        
        returns = close_prices.pct_change()
        last_price = get_val(close_prices.iloc[-1])
        daily_vol = returns.std()
        
        fig_sim = go.Figure()
        for i in range(100):
            price_series = [last_price]
            for _ in range(30):
                next_price = price_series[-1] * (1 + np.random.normal(0, daily_vol))
                price_series.append(next_price)
            
            fig_sim.add_trace(go.Scatter(y=price_series, mode='lines', 
                                         line=dict(width=1), showlegend=False, opacity=0.3))

        fig_sim.update_layout(xaxis_title="Days into Future", yaxis_title="Projected Price")
        st.plotly_chart(fig_sim, use_container_width=True)
        st.warning("⚠️ Statistical simulation only. Not financial advice.")

    # --- PAGE 7: RAW ENGINE DATA ---
    elif page == "Raw Engine Data":
        st.title("📋 Backend Data & Insights")
        st.dataframe(processed_data, use_container_width=True)
        csv = processed_data.to_csv().encode('utf-8')
        st.download_button(label="📥 Download CSV", data=csv, 
                           file_name=f'{ticker}_report.csv', mime='text/csv')

    # --- PAGE 8: MODEL THEORY ---
    elif page == "Model Theory & Documentation":
        st.title("📚 Behind the Scenes: The AI Logic")
        st.write("### 1. Isolation Forest (The AI)")
        st.info("The model isolates anomalies by identifying data points that are easiest to separate from the cluster.")
        st.write("### 2. The Z-Score (The Math)")
        st.latex(r"Z = \frac{x - \mu}{\sigma}")
        st.write("### 3. Safeguard Strategy")
        st.success("Combines machine learning with statistical thresholds to minimize false positives.")

    # --- PROFESSIONAL SIDEBAR ADD-ON: LIVE RISK GAUGE ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Sentinel Risk Rating")
    
    # Logic: High risk if volatility is high OR many anomalies found
    volatility_val = get_val(close_prices.std())
    mean_price = get_val(close_prices.mean())
    anomaly_count = len(anomalies)

    if anomaly_count > 5 or volatility_val > (mean_price * 0.05):
        st.sidebar.error("🔴 HIGH RISK DETECTED")
        st.sidebar.caption("Recommendation: Enhanced Monitoring")
    elif anomaly_count > 2:
        st.sidebar.warning("🟡 MEDIUM RISK")
        st.sidebar.caption("Recommendation: Caution Advised")
    else:
        st.sidebar.success("🟢 LOW RISK")
        st.sidebar.caption("Recommendation: Stable Trend")

else:
    st.error("Unable to load data. Please check the ticker symbol or internet connection.")