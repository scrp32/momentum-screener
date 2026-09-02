import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Page Layout Configuration
st.set_page_config(
    page_title="Institutional Momentum Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Institutional Momentum Screener (NSE / BSE)")
st.caption(
    "Volatility-Adjusted & Residual Momentum Engine with Real-Time Filtering"
)


# Sidebar Configuration & Filters
st.sidebar.header("🔍 Screener Filters")

# Sector / Industry Selection or Pre-built Lists
universe_type = st.sidebar.radio(
    "Select Stock Universe:",
    ["Nifty 50", "Nifty Next 50", "Nifty Midcap 100", "Custom List"],
)

# Benchmark Selection
benchmark_ticker = st.sidebar.selectbox(
    "Benchmark Index (for Beta/Residual Calculation):",
    ["^NSEI", "^BSESN"],
    format_func=lambda x: "Nifty 50 (^NSEI)"
    if x == "^NSEI"
    else "Sensex (^BSESN)",
)

st.sidebar.subheader("Quantitative Metric Thresholds")

# Sliders for Quantitative Filters
min_score, max_score = st.sidebar.slider(
    "Volatility-Adjusted Score Range:", -3.0, 5.0, (0.5, 4.0), step=0.1
)

min_beta, max_beta = st.sidebar.slider(
    "Beta Range (vs Benchmark):", 0.0, 3.0, (0.5, 1.8), step=0.1
)

max_volatility = st.sidebar.slider(
    "Max Annualized Volatility (%):", 10, 100, 45, step=5
)

min_residual_momentum = st.sidebar.slider(
    "Min Residual Momentum (%):", -50, 100, 0, step=5
)


# Helper function to get tickers based on selection
@st.cache_data(ttl=86400)
def get_universe_tickers(universe_name):
    if universe_name == "Nifty 50":
        return [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "TATAMOTORS.NS",
            "BHARTIARTL.NS",
            "ITC.NS",
            "LT.NS",
            "SBIN.NS",
            "AXISBANK.NS",
            "KOTAKBANK.NS",
            "HINDUNILVR.NS",
            "BAJFINANCE.NS",
            "MARUTI.NS",
            "SUNPHARMA.NS",
            "TITAN.NS",
            "ULTRACEMCO.NS",
            "ASIANPAINT.NS",
            "NTPC.NS",
        ]
    elif universe_name == "Nifty Next 50":
        return [
            "BEL.NS",
            "HAL.NS",
            "TRENT.NS",
            "VBL.NS",
            "ZOMATO.NS",
            "DLF.NS",
            "IOC.NS",
            "REC.NS",
            "PFC.NS",
            "BANKBARODA.NS",
        ]
    elif universe_name == "Nifty Midcap 100":
        return [
            "COALINDIA.NS",
            "NMDC.NS",
            "SAIL.NS",
            "IRFC.NS",
            "RVNL.NS",
            "POLYCAB.NS",
            "PERSISTENT.NS",
            "MPHASIS.NS",
        ]
    else:
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]


# Quant Computation Pipeline
@st.cache_data(ttl=3600)
def compute_quant_momentum(tickers, benchmark):
    all_tickers = tickers + [benchmark]
    data = yf.download(all_tickers, period="1y", interval="1d", progress=False)[
        "Close"
    ]

    bench_returns = data[benchmark].pct_change().dropna()
    bench_total_return = (
        data[benchmark].iloc[-1] - data[benchmark].iloc[0]
    ) / data[benchmark].iloc[0]

    results = []
    tickers_to_process = [t for t in tickers if t in data.columns]

    for ticker in tickers_to_process:
        stock_series = data[ticker].dropna()
        if len(stock_series) < 180:
            continue

        daily_returns = stock_series.pct_change().dropna()

        # Fetch market cap dynamically
        try:
            info = yf.Ticker(ticker).fast_info
            mcap_cr = round(info.market_cap / 1e7, 2)  # Convert to INR Crores
        except Exception:
            mcap_cr = np.nan

        # 12M-1M Cross-Sectional Momentum
        price_12m_ago = stock_series.iloc[0]
        price_1m_ago = stock_series.iloc[-21]
        raw_12m_1m_return = (
            price_1m_ago - price_12m_ago
        ) / price_12m_ago

        # Volatility Scaling
        annualized_vol = daily_returns.std() * np.sqrt(252)
        vol_adjusted_score = (
            (raw_12m_1m_return / annualized_vol) if annualized_vol > 0 else 0
        )

        # Residual Return (Beta Isolation)
        aligned = pd.concat(
            [daily_returns, bench_returns], axis=1, join="inner"
        ).dropna()
        aligned.columns = ["stock", "bench"]
        cov = np.cov(aligned["stock"], aligned["bench"])[0][1]
        bench_var = np.var(aligned["bench"])
        beta = cov / bench_var if bench_var > 0 else 1.0

        stock_total_return = (
            stock_series.iloc[-1] - stock_series.iloc[0]
        ) / stock_series.iloc[0]
        residual_momentum = stock_total_return - (beta * bench_total_return)

        results.append(
            {
                "Ticker": ticker.replace(".NS", "").replace(".BO", ""),
                "Market Cap (Cr)": mcap_cr,
                "12M-1M Return (%)": round(raw_12m_1m_return * 100, 2),
                "Annual Volatility (%)": round(annualized_vol * 100, 2),
                "Vol-Adjusted Score": round(vol_adjusted_score, 2),
                "Beta": round(beta, 2),
                "Residual Return (%)": round(residual_momentum * 100, 2),
            }
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df["Rank"] = (
            df["Vol-Adjusted Score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        df = df.sort_values(by="Rank").reset_index(drop=True)
    return df


# Execute Engine
tickers_list = get_universe_tickers(universe_type)

with st.spinner("Processing stock universe and running quantitative models..."):
    df_raw = compute_quant_momentum(tickers_list, benchmark_ticker)

if not df_raw.empty:
    # Additional Sidebar Filter for Market Cap
    min_mcap = int(df_raw["Market Cap (Cr)"].min(skipna=True))
    max_mcap = int(df_raw["Market Cap (Cr)"].max(skipna=True))

    mcap_range = st.sidebar.slider(
        "Market Cap Range (₹ Crores):",
        min_value=min_mcap,
        max_value=max_mcap,
        value=(min_mcap, max_mcap),
    )

    # Filtering Logic
    filtered_df = df_raw[
        (df_raw["Vol-Adjusted Score"] >= min_score)
        & (df_raw["Vol-Adjusted Score"] <= max_score)
        & (df_raw["Beta"] >= min_beta)
        & (df_raw["Beta"] <= max_beta)
        & (df_raw["Annual Volatility (%)"] <= max_volatility)
        & (df_raw["Residual Return (%)"] >= min_residual_momentum)
        & (
            df_raw["Market Cap (Cr)"].isna()
            | (
                (df_raw["Market Cap (Cr)"] >= mcap_range[0])
                & (df_raw["Market Cap (Cr)"] <= mcap_range[1])
            )
        )
    ]

    # Executive Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Universe Stocks", len(df_raw))
    col2.metric("Filtered Candidates", len(filtered_df))
    col3.metric(
        "Avg Vol-Adjusted Score",
        f"{filtered_df['Vol-Adjusted Score'].mean():.2f}"
        if not filtered_df.empty
        else "N/A",
    )
    col4.metric(
        "Avg Residual Return",
        f"{filtered_df['Residual Return (%)'].mean():.2f}%"
        if not filtered_df.empty
        else "N/A",
    )

    st.divider()

    # Data Table View
    st.subheader("📋 Quant Momentum Screener Table")
    st.dataframe(
        filtered_df.style.highlight_max(
            axis=0, subset=["Vol-Adjusted Score", "Residual Return (%)"]
        ),
        use_container_width=True,
    )

    # Plotly Scatter Chart: Risk vs Momentum Profile
    st.subheader("📊 Momentum Factor Matrix")
    if not filtered_df.empty:
        fig = px.scatter(
            filtered_df,
            x="Annual Volatility (%)",
            y="Vol-Adjusted Score",
            size="Market Cap (Cr)",
            color="Residual Return (%)",
            hover_name="Ticker",
            title="Volatility vs Vol-Adjusted Momentum (Bubble Size = Market Cap)",
            labels={
                "Annual Volatility (%)": "Annualized Risk (%)",
                "Vol-Adjusted Score": "Sharpe-Scaled Momentum",
            },
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("No data fetched. Check your internet connection or ticker universe.")