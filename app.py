from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Page setup
st.set_page_config(
    page_title="Institutional Momentum Screener & Paper Trader",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# PASSWORD PROTECTION FUNCTION
# -----------------------------------------------------------------------------
def check_password():
    def password_entered():
        if st.session_state["password_input"] == "quant123":
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Institutional Momentum Screener")
        st.text_input(
            "Enter Password to Access Dashboard:",
            type="password",
            on_change=password_entered,
            key="password_input",
        )
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Institutional Momentum Screener")
        st.text_input(
            "Enter Password to Access Dashboard:",
            type="password",
            on_change=password_entered,
            key="password_input",
        )
        st.error("😕 Incorrect Password. Please try again.")
        return False
    else:
        return True


# -----------------------------------------------------------------------------
# PAPER TRADING SIMULATION ENGINE WITH FORWARD EXECUTION
# -----------------------------------------------------------------------------
def simulate_paper_trade(
    price_series,
    entry_date,
    capital_per_trade=100000,
    target_pct=0.10,
    stop_loss_pct=0.05,
):
    """Simulates paper trading starting from a specific entry_date forward.

    Allocates fixed capital (₹1 Lakh) and tracks status, days held, trailing
    stop, and PnL.
    """
    price_series.index = pd.to_datetime(price_series.index)
    entry_dt = pd.to_datetime(entry_date)

    # Filter price series from entry date onwards
    forward_prices = price_series[price_series.index >= entry_dt]

    # If entry date is in the future (e.g. Tomorrow), mark as PENDING
    if forward_prices.empty or entry_dt > pd.to_datetime(datetime.now().date()):
        latest_price = price_series.iloc[-1]
        return {
            "Entry Date": entry_dt.strftime("%Y-%m-%d"),
            "Entry Price (INR)": round(latest_price, 2),
            "Paper Status": "PENDING (Scheduled)",
            "Days Held": 0,
            "Exit Reason": "Awaiting Entry",
            "Exit Date": "N/A",
            "Return (%)": 0.0,
            "PnL (INR)": 0.0,
            "Current Trailing Stop": round(
                latest_price * (1 - stop_loss_pct), 2
            ),
        }

    entry_price = forward_prices.iloc[0]
    initial_stop = entry_price * (1 - stop_loss_pct)
    target_price = entry_price * (1 + target_pct)

    current_stop = initial_stop
    peak_price = entry_price

    status = "OPEN"
    exit_reason = "Open Position"
    exit_price = forward_prices.iloc[-1]
    exit_date_str = "Active"
    days_held = len(forward_prices)

    for idx, (dt, price) in enumerate(forward_prices.items()):
        # Trail stop-loss if new peak is formed
        if price > peak_price:
            peak_price = price
            current_stop = max(current_stop, peak_price * (1 - stop_loss_pct))

        # Condition 1: Take Profit (+10%)
        if price >= target_price:
            status = "CLOSED"
            exit_reason = "Target Hit (+10%)"
            exit_price = target_price
            exit_date_str = dt.strftime("%Y-%m-%d")
            days_held = idx + 1
            break

        # Condition 2: Trailing Stop Hit (-5% from peak)
        elif price <= current_stop:
            status = "CLOSED"
            exit_reason = "Trailing Stop Hit"
            exit_price = current_stop
            exit_date_str = dt.strftime("%Y-%m-%d")
            days_held = idx + 1
            break

    return_pct = ((exit_price - entry_price) / entry_price) * 100
    pnl_inr = capital_per_trade * (return_pct / 100)

    return {
        "Entry Date": entry_dt.strftime("%Y-%m-%d"),
        "Entry Price (INR)": round(entry_price, 2),
        "Paper Status": status,
        "Days Held": days_held,
        "Exit Reason": exit_reason,
        "Exit Date": exit_date_str,
        "Return (%)": round(return_pct, 2),
        "PnL (INR)": round(pnl_inr, 2),
        "Current Trailing Stop": round(current_stop, 2),
    }


# -----------------------------------------------------------------------------
# MAIN APP EXECUTION
# -----------------------------------------------------------------------------
if check_password():

    st.title("📈 Institutional Momentum Screener & Paper Trader")
    st.caption("Volatility-Adjusted Momentum Engine with Live Paper Execution")

    st.sidebar.header("🔍 Universe & Schedule Settings")

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

    # Universe Selection Dropdown
    universe_type = st.sidebar.selectbox(
        "Select Stock Universe:",
        [
            "Nifty 50",
            "Nifty 100",
            "Nifty 500",
            "Nifty Microcap 250",
            "Nifty Total Market (~750)",
        ],
        index=4,  # Default to Nifty Total Market (~750)
    )

    benchmark_ticker = st.sidebar.selectbox(
        "Benchmark Index (for Beta/Residual Calculation):",
        ["^NSEI", "^BSESN"],
        format_func=lambda x: (
            "Nifty 50 (^NSEI)" if x == "^NSEI" else "Sensex (^BSESN)"
        ),
    )

    # Entry Date Stagger Settings
    tomorrow_date = datetime.now().date() + timedelta(days=1)
    base_start_date = st.sidebar.date_input(
        "First Candidate Entry Date:", tomorrow_date
    )
    stagger_days = st.sidebar.number_input(
        "Stagger Days Between Subsequent Additions:",
        min_value=0,
        max_value=10,
        value=1,
        help="Trading days delay before adding the next candidate stock to paper portfolio.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Quantitative Metric Filters")

    min_score, max_score = st.sidebar.slider(
        "Volatility-Adjusted Score Range:", -3.0, 5.0, (0.5, 4.0), step=0.1
    )

    max_dma_extension = st.sidebar.slider(
        "Max 200 DMA Extension (%) [Exhaustion Guardrail]:",
        10,
        100,
        30,
        step=5,
    )

    min_beta, max_beta = st.sidebar.slider(
        "Beta Range (vs Benchmark):", 0.0, 3.0, (0.5, 1.8), step=0.1
    )

    max_volatility = st.sidebar.slider(
        "Max Annual Volatility (%):", 10, 100, 45, step=5
    )

    min_residual_momentum = st.sidebar.slider(
        "Min Residual Return (%):", -50, 100, 0, step=5
    )

    # Computation Engine
    @st.cache_data(ttl=3600)
    def compute_quant_momentum(tickers, benchmark):
        all_tickers = tickers + [benchmark]
        raw_data = yf.download(
            all_tickers, period="1y", interval="1d", progress=False
        )

        data = (
            raw_data["Close"]
            if isinstance(raw_data.columns, pd.MultiIndex)
            else raw_data
        )

        bench_returns = data[benchmark].pct_change(fill_method=None).dropna()
        bench_total_return = (
            data[benchmark].iloc[-1] - data[benchmark].iloc[0]
        ) / data[benchmark].iloc[0]

        results = []
        tickers_to_process = [t for t in tickers if t in data.columns]

        for ticker in tickers_to_process:
            stock_series = data[ticker].dropna()
            if len(stock_series) < 200:
                continue

            daily_returns = stock_series.pct_change(fill_method=None).dropna()

            try:
                info = yf.Ticker(ticker).fast_info
                mcap_cr = round(info.market_cap / 1e7, 2)
            except Exception:
                mcap_cr = np.nan

            current_price = stock_series.iloc[-1]
            sma_200 = stock_series.rolling(window=200).mean().iloc[-1]
            dma_200_ext = (
                ((current_price - sma_200) / sma_200) * 100
                if sma_200 > 0
                else 0
            )

            price_12m_ago = stock_series.iloc[0]
            price_1m_ago = stock_series.iloc[-21]
            raw_12m_1m_return = (price_1m_ago - price_12m_ago) / price_12m_ago

            annualized_vol = daily_returns.std() * np.sqrt(252)
            vol_adjusted_score = (
                (raw_12m_1m_return / annualized_vol) if annualized_vol > 0 else 0
            )

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
            residual_momentum = stock_total_return - (
                beta * bench_total_return
            )

            results.append(
                {
                    "Ticker": ticker.replace(".NS", "").replace(".BO", ""),
                    "Market Cap (Cr)": mcap_cr,
                    "200 DMA Ext (%)": round(dma_200_ext, 2),
                    "12M-1M Return (%)": round(raw_12m_1m_return * 100, 2),
                    "Annual Volatility (%)": round(annualized_vol * 100, 2),
                    "Vol-Adjusted Score": round(vol_adjusted_score, 2),
                    "Beta": round(beta, 2),
                    "Residual Return (%)": round(residual_momentum * 100, 2),
                    "Price Series": stock_series,
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
        f"Processing Momentum Metrics & Paper Simulation for {len(tickers_list)} tickers in {universe_type}..."
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
            value=(
                min_mcap,
                max_mcap if max_mcap > min_mcap else min_mcap + 1000,
            ),
        )

        # Apply Screener Filters
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
        ].copy()

        # Run Paper Trade Simulation for Filtered Candidates
        sim_results = []
        for idx, row in filtered_df.reset_index(drop=True).iterrows():
            assigned_entry_date = base_start_date + timedelta(
                days=idx * stagger_days
            )
            sim = simulate_paper_trade(
                row["Price Series"],
                entry_date=assigned_entry_date,
                capital_per_trade=100000,
                target_pct=0.10,
                stop_loss_pct=0.05,
            )
            sim_results.append(sim)

        sim_df = pd.DataFrame(sim_results)

        display_df = pd.concat(
            [
                filtered_df.drop(columns=["Price Series"]).reset_index(
                    drop=True
                ),
                sim_df,
            ],
            axis=1,
        )

        # Summary Metrics
        total_allocated = len(display_df) * 100000
        total_pnl = (
            display_df["PnL (INR)"].sum() if not display_df.empty else 0
        )
        total_return_pct = (
            (total_pnl / total_allocated) * 100 if total_allocated > 0 else 0
        )
        avg_days = display_df["Days Held"].mean() if not display_df.empty else 0

        # Performance Metrics Header
        st.subheader("💵 Portfolio Overview & Paper Execution")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Qualified Candidates", len(display_df))
        m2.metric("Total Capital Allocated", f"₹{total_allocated:,.0f}")
        m3.metric(
            "Total Portfolio PnL",
            f"₹{total_pnl:,.2f}",
            delta=f"{total_return_pct:.2f}% Net Return",
        )
        m4.metric("Avg Days Held", f"{avg_days:.1f} Days")
        m5.metric(
            "Status Breakdown",
            f"{len(display_df[display_df['Paper Status'] == 'OPEN'])} Open / "
            f"{len(display_df[display_df['Paper Status'] == 'CLOSED'])} Closed / "
            f"{len(display_df[display_df['Paper Status'].str.contains('PENDING', na=False)])} Scheduled",
        )

        st.divider()

        # Quantitative Ledger Table
        st.subheader("📋 Momentum Screener & Staggered Execution Ledger")
        st.dataframe(
            display_df.style.highlight_max(
                axis=0, subset=["Vol-Adjusted Score", "Return (%)"]
            ),
            use_container_width=True,
        )

        # Visualization
        st.subheader("📊 Individual Trade Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            fig_pnl = px.bar(
                display_df,
                x="Ticker",
                y="PnL (INR)",
                color="Paper Status",
                title="PnL per Stock (₹1 Lakh Fixed Allocation)",
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col2:
            fig_days = px.bar(
                display_df,
                x="Ticker",
                y="Days Held",
                color="Exit Reason",
                title="Holding Duration per Ticker (Days)",
            )
            st.plotly_chart(fig_days, use_container_width=True)
    else:
        st.error(
            "No stocks met the current criteria. Try loosening your filter"
            " threshold ranges in the sidebar."
        )
