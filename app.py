import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Page setup
st.set_page_config(
    page_title="Institutional Momentum Screener",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

st.title("📈 Institutional Momentum Screener (NSE / BSE)")
st.caption(
    "Volatility-Adjusted & Residual Momentum Engine with Parabolic Guardrails"
)

st.sidebar.header("🔍 Screener Filters")


# Fetch Universe Lists dynamically from NSE archives with 24h caching
@st.cache_data(ttl=86400)
def fetch_universe_tickers(universe_name):
    urls = {
        "Nifty 50": (
            "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        ),
        "Nifty 100": (
            "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
        ),
        "Nifty 500": (
            "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        ),
        "Nifty Microcap 250": (
            "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv"
        ),
        "Nifty Total Market (~750)": (
            "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
        ),
    }
    url = urls.get(universe_name)
    if not url:
        return [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
        ]

    try:
        df = pd.read_csv(url)
        symbols = df["Symbol"].dropna().str.strip().tolist()
        return [f"{symbol}.NS" for symbol in symbols]
    except Exception:
        return [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
        ]


# Universe Selection
universe_type = st.sidebar.selectbox(
    "Select Stock Universe:",
    [
        "Nifty 50",
        "Nifty 100",
        "Nifty 500",
        "Nifty Microcap 250",
        "Nifty Total Market (~750)",
    ],
    index=2,  # Default to Nifty 500
)

benchmark_ticker = st.sidebar.selectbox(
    "Benchmark Index (for Beta/Residual Calculation):",
    ["^NSEI", "^BSESN"],
    format_func=lambda x: (
        "Nifty 50 (^NSEI)" if x == "^NSEI" else "Sensex (^BSESN)"
    ),
)

st.sidebar.subheader("Quantitative Metric Thresholds")

min_score, max_score = st.sidebar.slider(
    "Volatility-Adjusted Score Range:", -3.0, 5.0, (0.5, 4.0), step=0.1
)

# 🛑 Exhaustion Safeguard: 200 DMA Extension Slider
max_dma_extension = st.sidebar.slider(
    "Max 200 DMA Extension (%) [Exhaustion Guardrail]:",
    10,
    100,
    30,
    step=5,
    help=(
        "Filters out stocks trading too far above their 200-day moving average"
        " (prevents buying parabolic peaks)."
    ),
)

min_beta, max_beta = st.sidebar.slider(
    "Beta Range (vs Benchmark):", 0.0, 3.0, (0.5, 1.8), step=0.1
)

max_volatility = st.sidebar.slider(
    "Max Annual Volatility (%):", 10, 100, 45, step=5
)

min_residual_momentum = st.sidebar.slider(
    "Min Residual Momentum (%):", -50, 100, 0, step=5
)


# Fetch and compute price data with caching
@st.cache_data(ttl=3600)
def compute_quant_momentum(tickers, benchmark):
    all_tickers = tickers + [benchmark]
    raw_data = yf.download(
        all_tickers, period="1y", interval="1d", progress=False
    )

    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data["Close"]
    else:
        data = raw_data

    bench_returns = data[benchmark].pct_change(fill_method=None).dropna()
    bench_total_return = (
        data[benchmark].iloc[-1] - data[benchmark].iloc[0]
    ) / data[benchmark].iloc[0]

    results = []
    tickers_to_process = [t for t in tickers if t in data.columns]

    for ticker in tickers_to_process:
        stock_series = data[ticker].dropna()
        if len(stock_series) < 200:  # Need at least 200 days for 200 DMA
            continue

        daily_returns = stock_series.pct_change(fill_method=None).dropna()

        try:
            info = yf.Ticker(ticker).fast_info
            mcap_cr = round(info.market_cap / 1e7, 2)
        except Exception:
            mcap_cr = np.nan

        current_price = stock_series.iloc[-1]

        # 200-Day Simple Moving Average & Extension Math
        sma_200 = stock_series.rolling(window=200).mean().iloc[-1]
        dma_200_extension = (
            ((current_price - sma_200) / sma_200) * 100 if sma_200 > 0 else 0
        )

        # 12M - 1M Cross-Sectional Return
        price_12m_ago = stock_series.iloc[0]
        price_1m_ago = stock_series.iloc[-21]
        raw_12m_1m_return = (price_1m_ago - price_12m_ago) / price_12m_ago

        annualized_vol = daily_returns.std() * np.sqrt(252)
        vol_adjusted_score = (
            (raw_12m_1m_return / annualized_vol) if annualized_vol > 0 else 0
        )

        # Beta & Residual Momentum
        aligned = pd.concat(
            [daily_returns, bench_returns], axis=1, join="inner"
        ).dropna()
        aligned.columns = ["stock", "bench"]
        cov = np.cov(aligned["stock"], aligned["bench"])[0][1]
        bench_var = np.var(aligned["bench"])
        beta = cov / bench_var if bench_var > 0 else 1.0

        stock_total_return = (
            current_price - stock_series.iloc[0]
        ) / stock_series.iloc[0]
        residual_momentum = stock_total_return - (beta * bench_total_return)

        results.append(
            {
                "Ticker": ticker.replace(".NS", "").replace(".BO", ""),
                "Market Cap (Cr)": mcap_cr,
                "200 DMA Ext (%)": round(dma_200_extension, 2),
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


tickers_list = fetch_universe_tickers(universe_type)

with st.spinner(
    f"Fetching data for {len(tickers_list)} stocks in {universe_type}..."
):
    df_raw = compute_quant_momentum(tickers_list, benchmark_ticker)

if not df_raw.empty:
    min_mcap = (
        int(df_raw["Market Cap (Cr)"].min(skipna=True))
        if not df_raw["Market Cap (Cr)"].isna().all()
        else 0
    )
    max_mcap = (
        int(df_raw["Market Cap (Cr)"].max(skipna=True))
        if not df_raw["Market Cap (Cr)"].isna().all()
        else 100000
    )

    mcap_range = st.sidebar.slider(
        "Market Cap Range (INR Crores):",
        min_value=min_mcap,
        max_value=max_mcap if max_mcap > min_mcap else min_mcap + 1000,
        value=(min_mcap, max_mcap if max_mcap > min_mcap else min_mcap + 1000),
    )

    # Filter pipeline including 200 DMA Extension
    filtered_df = df_raw[
        (df_raw["Vol-Adjusted Score"] >= min_score)
        & (df_raw["Vol-Adjusted Score"] <= max_score)
        & (df_raw["200 DMA Ext (%)"] <= max_dma_extension)
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Universe Stocks", len(df_raw))
    col2.metric("Filtered Candidates", len(filtered_df))
    col3.metric(
        "Avg Vol-Adjusted Score",
        (
            f"{filtered_df['Vol-Adjusted Score'].mean():.2f}"
            if not filtered_df.empty
            else "N/A"
        ),
    )
    col4.metric(
        "Avg Residual Return",
        (
            f"{filtered_df['Residual Return (%)'].mean():.2f}%"
            if not filtered_df.empty
            else "N/A"
        ),
    )

    st.divider()

    st.subheader("📋 Quant Momentum Screener Table")
    st.dataframe(
        filtered_df.style.highlight_max(
            axis=0, subset=["Vol-Adjusted Score", "Residual Return (%)"]
        ),
        use_container_width=True,
    )

    st.subheader("📊 Momentum Factor Matrix")
    if not filtered_df.empty:
        fig = px.scatter(
            filtered_df,
            x="200 DMA Ext (%)",
            y="Vol-Adjusted Score",
            size="Market Cap (Cr)",
            color="Residual Return (%)",
            hover_name="Ticker",
            title="200 DMA Extension vs Vol-Adjusted Momentum Score",
            labels={
                "200 DMA Ext (%)": "Extension Above 200 DMA (%)",
                "Vol-Adjusted Score": "Sharpe-Scaled Momentum",
            },
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error(
        "No data fetched. Check your internet connection or ticker universe."
    )
