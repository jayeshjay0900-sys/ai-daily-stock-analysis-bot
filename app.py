from __future__ import annotations

import html
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.fundamental import get_fundamentals
from analysis.risk import calculate_risk
from analysis.sentiment import analyze_news_sentiment, sentiment_summary
from analysis.technical import analyze_technicals
from config import NEWS_LIMIT
from models.prediction_model import run_direction_model
from services.market_data import get_history
from services.news import fetch_news


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Daily Stock Analysis",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .small-note {
        font-size: 13px;
        opacity: 0.75;
    }

    .table-container {
        width: 100%;
        max-height: 500px;
        overflow-x: auto;
        overflow-y: auto;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .stock-table {
        width: 100%;
        min-width: 650px;
        border-collapse: collapse;
        font-size: 13px;
    }

    .stock-table th {
        position: sticky;
        top: 0;
        z-index: 2;
        padding: 8px 6px;
        text-align: center;
        background-color: #262730;
        color: white;
        border-bottom: 2px solid #777;
        white-space: nowrap;
    }

    .stock-table td {
        padding: 7px 6px;
        text-align: center;
        border-bottom: 1px solid rgba(128,128,128,0.25);
        white-space: nowrap;
    }

    .stock-table tr:hover {
        background-color: rgba(128,128,128,0.10);
    }

    .stock-table a {
        text-decoration: none;
        font-weight: bold;
    }

    .stock-table a:hover {
        text-decoration: underline;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CACHE — MARKET DATA
# ============================================================

@st.cache_data(ttl=900)
def load_stock_data(
    ticker: str,
    period: str,
    refresh_token: int,
):
    return get_history(
        ticker,
        period=period,
        refresh=True,
    )


# ============================================================
# CACHE — NEWS
# ============================================================

@st.cache_data(ttl=3600)
def load_news(
    ticker: str,
    company_name: str,
):
    articles = fetch_news(
        ticker,
        company_name,
        NEWS_LIMIT,
    )

    return analyze_news_sentiment(
        articles
    )


# ============================================================
# FORMAT NUMBER
# ============================================================

def fmt(
    value,
    suffix="",
):
    if value is None:
        return "N/A"

    try:

        if pd.isna(value):
            return "N/A"

        return f"{float(value):,.2f}{suffix}"

    except Exception:

        return str(value)


# ============================================================
# SAFE VALUE
# ============================================================

def safe_value(
    data,
    key,
    default=None,
):
    """
    Safely retrieve dictionary values.
    """

    if not isinstance(data, dict):
        return default

    value = data.get(key, default)

    if value is None:
        return default

    return value


# ============================================================
# HTML TABLE
# ============================================================

def display_dataframe(
    df: pd.DataFrame,
    min_width=650,
):
    """
    Display DataFrame using HTML instead of st.dataframe().

    This avoids the PyArrow DLL problem.
    """

    if df is None:
        st.info("No data available.")
        return

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            st.info("No data available.")
            return

    if df.empty:
        st.info("No data available.")
        return

    # Make a copy so original data is not modified.
    display_df = df.copy()

    # Convert problematic values to display-safe strings.
    for column in display_df.columns:

        display_df[column] = display_df[column].apply(
            lambda x: (
                "N/A"
                if x is None
                or (
                    not isinstance(x, (dict, list, tuple))
                    and pd.isna(x)
                )
                else str(x)
            )
        )

    table_html = display_df.to_html(
        index=True,
        border=0,
        classes="stock-table",
        escape=False,
        justify="center",
    )

    full_html = f"""
    <div
        class="table-container"
        style="min-width: 100%;"
    >
        {table_html}
    </div>
    """

    st.html(full_html)


# ============================================================
# SAFE HTML LINK
# ============================================================

def make_open_link(
    url,
):
    """
    Convert a long URL into a small Open link.
    """

    if url is None:
        return "N/A"

    try:

        if pd.isna(url):
            return "N/A"

    except Exception:
        pass

    url = str(url).strip()

    if not url:
        return "N/A"

    # Basic protection against malformed URLs.
    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):
        return "Open"

    safe_url = html.escape(
        url,
        quote=True,
    )

    return (
        f'<a href="{safe_url}" '
        f'target="_blank">Open</a>'
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 AI Daily Stock Market Analysis Bot"
)

st.caption(
    "Market data + technical analysis + risk + "
    "news sentiment + historical ML direction classification"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Settings"
    )

    ticker = st.text_input(
        "Stock ticker",
        value="AAPL",
        help=(
            "Examples: AAPL, MSFT, "
            "RELIANCE.NS, TCS.NS, INFY.NS"
        ),
    ).strip().upper()

    period = st.selectbox(
        "Historical period",
        [
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
        ],
        index=1,
    )

    refresh = st.button(
        "🔄 Refresh market data",
        use_container_width=True,
    )

    st.divider()

    st.info(
        "Data may be delayed depending on the provider. "
        "This application is for analysis/education and "
        "is not financial advice."
    )


# ============================================================
# CHECK TICKER
# ============================================================

if not ticker:

    st.warning(
        "Enter a ticker symbol."
    )

    st.stop()


# ============================================================
# LOAD MAIN DATA
# ============================================================

try:

    with st.spinner(
        f"Loading {ticker}..."
    ):

        df = load_stock_data(
            ticker,
            period,
            int(refresh),
        )

    if df is None or df.empty:

        st.error(
            f"No historical data was returned for {ticker}."
        )

        st.stop()

    # --------------------------------------------------------
    # TECHNICAL ANALYSIS
    # --------------------------------------------------------

    technical = analyze_technicals(
        df
    )

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    benchmark_ticker = (
        "^NSEI"
        if ticker.endswith(".NS")
        else "^GSPC"
    )

    try:

        benchmark = get_history(
            benchmark_ticker,
            period=period,
            refresh=bool(refresh),
        )

    except Exception:

        benchmark = None

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    try:

        risk = calculate_risk(
            df,
            benchmark,
        )

    except Exception as risk_error:

        risk = {}

        st.sidebar.warning(
            f"Risk calculation issue: {risk_error}"
        )

    # --------------------------------------------------------
    # FUNDAMENTALS
    # --------------------------------------------------------

    try:

        fundamentals = get_fundamentals(
            ticker
        )

    except Exception as fundamental_error:

        fundamentals = {}

        st.sidebar.warning(
            f"Fundamental data issue: "
            f"{fundamental_error}"
        )

    if not isinstance(
        fundamentals,
        dict,
    ):

        fundamentals = {}

    company_name = (
        fundamentals.get("longName")
        or fundamentals.get("shortName")
        or ticker
    )


except Exception as exc:

    st.error(
        f"Unable to load {ticker}: {exc}"
    )

    st.stop()


# ============================================================
# LATEST MARKET DATA
# ============================================================

try:

    latest_date = (
        df.index[-1].strftime(
            "%Y-%m-%d"
        )
    )

except Exception:

    latest_date = "N/A"


try:

    latest_price = float(
        df["Close"].iloc[-1]
    )

except Exception:

    latest_price = 0.0


try:

    previous_price = float(
        df["Close"].iloc[-2]
    )

except Exception:

    previous_price = latest_price


if previous_price:

    daily_change = (
        (
            latest_price
            / previous_price
            - 1
        )
        * 100
    )

else:

    daily_change = 0


# ============================================================
# HEADER
# ============================================================

st.subheader(
    f"📊 {company_name} ({ticker})"
)

st.caption(
    f"Latest available market-data date: "
    f"{latest_date}"
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4, c5, c6 = st.columns(6)


c1.metric(
    "Price",
    fmt(latest_price),
)


c2.metric(
    "Daily Change",
    f"{daily_change:.2f}%",
)


c3.metric(
    "RSI",
    fmt(
        safe_value(
            technical,
            "rsi",
        )
    ),
)


volume_value = safe_value(
    technical,
    "volume",
)

try:

    volume_text = f"{float(volume_value):,.0f}"

except Exception:

    volume_text = "N/A"


c4.metric(
    "Volume",
    volume_text,
)


volatility = safe_value(
    risk,
    "annualized_volatility",
)

try:

    volatility_text = fmt(
        float(volatility) * 100,
        "%",
    )

except Exception:

    volatility_text = "N/A"


c5.metric(
    "Volatility",
    volatility_text,
)


c6.metric(
    "News",
    "Load below",
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Price",
        "📉 Technical",
        "📰 News",
        "⚠️ Risk",
        "🤖 ML",
        "🏢 Fundamentals",
    ]
)


# ============================================================
# TAB 1 — PRICE
# ============================================================

with tab1:

    st.subheader(
        "Price chart"
    )

    try:

        fig = go.Figure()

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )

        tech_data = technical["data"]

        # SMA 20
        if "SMA_20" in tech_data.columns:

            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data["SMA_20"],
                    name="SMA 20",
                )
            )

        # SMA 50
        if "SMA_50" in tech_data.columns:

            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data["SMA_50"],
                    name="SMA 50",
                )
            )

        # SMA 200
        if "SMA_200" in tech_data.columns:

            fig.add_trace(
                go.Scatter(
                    x=tech_data.index,
                    y=tech_data["SMA_200"],
                    name="SMA 200",
                )
            )

        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    except Exception as chart_error:

        st.error(
            f"Unable to create price chart: "
            f"{chart_error}"
        )


    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Performance"
    )

    performance_periods = {
        "1D": 1,
        "5D": 5,
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }

    perf = {}

    for label, n in performance_periods.items():

        try:

            if len(df) > n:

                old_price = float(
                    df["Close"].iloc[-n - 1]
                )

                perf[label] = (
                    (
                        latest_price
                        / old_price
                        - 1
                    )
                    * 100
                )

            else:

                perf[label] = None

        except Exception:

            perf[label] = None


    cols = st.columns(6)

    for col, (
        label,
        value,
    ) in zip(
        cols,
        perf.items(),
    ):

        col.metric(
            label,
            (
                "N/A"
                if value is None
                else f"{value:.2f}%"
            ),
        )


# ============================================================
# TAB 2 — TECHNICAL
# ============================================================

with tab2:

    st.subheader(
        "Technical indicators"
    )

    try:

        indicator_cols = [
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "EMA_20",
            "EMA_50",
            "RSI_14",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower",
            "ATR_14",
        ]

        available_indicator_cols = [
            col
            for col in indicator_cols
            if col in technical["data"].columns
        ]

        if available_indicator_cols:

            indicator_table = (
                technical["data"]
                [available_indicator_cols]
                .tail(30)
                .round(4)
            )

            display_dataframe(
                indicator_table
            )

        else:

            st.warning(
                "No technical indicator columns "
                "are available."
            )


        st.write(
            "**Long-term:**",
            safe_value(
                technical,
                "long_term",
                "N/A",
            ),
        )

        st.write(
            "**Momentum:**",
            safe_value(
                technical,
                "momentum",
                "N/A",
            ),
        )

        st.write(
            "**MACD:**",
            safe_value(
                technical,
                "macd_interpretation",
                "N/A",
            ),
        )


        support = safe_value(
            technical,
            "support",
        )

        resistance = safe_value(
            technical,
            "resistance",
        )

        st.write(
            f"Recent support: "
            f"**{fmt(support)}** | "
            f"Recent resistance: "
            f"**{fmt(resistance)}**"
        )

    except Exception as technical_error:

        st.error(
            f"Technical analysis error: "
            f"{technical_error}"
        )


# ============================================================
# TAB 3 — NEWS
# ============================================================

with tab3:

    st.subheader(
        "News & sentiment"
    )

    try:

        articles = load_news(
            ticker,
            company_name,
        )

    except Exception as news_error:

        articles = []

        st.warning(
            f"News unavailable: "
            f"{news_error}"
        )


    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    try:

        summary = sentiment_summary(
            articles
        )

    except Exception:

        summary = {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        }


    n1, n2, n3 = st.columns(3)


    n1.metric(
        "Positive",
        f"{summary.get('positive', 0):.1f}%",
    )


    n2.metric(
        "Neutral",
        f"{summary.get('neutral', 0):.1f}%",
    )


    n3.metric(
        "Negative",
        f"{summary.get('negative', 0):.1f}%",
    )


    # --------------------------------------------------------
    # NEWS TABLE
    # --------------------------------------------------------

    if articles:

        try:

            news_df = pd.DataFrame(
                articles
            )

            news_display = pd.DataFrame()


            if "headline" in news_df.columns:

                news_display["Headline"] = (
                    news_df["headline"]
                )


            if "source" in news_df.columns:

                news_display["Source"] = (
                    news_df["source"]
                )


            if "published" in news_df.columns:

                news_display["Published"] = (
                    news_df["published"]
                )


            if "sentiment" in news_df.columns:

                news_display["Sentiment"] = (
                    news_df["sentiment"]
                )


            if "confidence" in news_df.columns:

                news_display["Confidence"] = (

                    pd.to_numeric(
                        news_df["confidence"],
                        errors="coerce",
                    )
                    .round(3)
                )


            if "url" in news_df.columns:

                news_display["URL"] = (
                    news_df["url"].apply(
                        make_open_link
                    )
                )


            display_dataframe(
                news_display
            )

        except Exception as news_table_error:

            st.error(
                f"Unable to display news: "
                f"{news_table_error}"
            )

    else:

        st.info(
            "No recent articles were retrieved."
        )


# ============================================================
# TAB 4 — RISK
# ============================================================

with tab4:

    st.subheader(
        "⚠️ Risk analysis"
    )

    try:

        # ----------------------------------------------------
        # GET RISK VALUES SAFELY
        # ----------------------------------------------------

        annualized_volatility = safe_value(
            risk,
            "annualized_volatility",
        )

        average_daily_return = safe_value(
            risk,
            "average_daily_return",
        )

        max_drawdown = safe_value(
            risk,
            "max_drawdown",
        )

        sharpe = safe_value(
            risk,
            "sharpe",
        )

        recent_volatility = safe_value(
            risk,
            "recent_volatility",
        )

        beta = safe_value(
            risk,
            "beta",
        )


        # ----------------------------------------------------
        # BUILD DISPLAY TABLE
        # ----------------------------------------------------

        risk_rows = [

            {
                "Metric": "Annualized volatility",
                "Value": (
                    float(annualized_volatility) * 100
                    if annualized_volatility is not None
                    else None
                ),
                "Unit": "%",
            },

            {
                "Metric": "Average daily return",
                "Value": (
                    float(average_daily_return) * 100
                    if average_daily_return is not None
                    else None
                ),
                "Unit": "%",
            },

            {
                "Metric": "Maximum drawdown",
                "Value": (
                    float(max_drawdown) * 100
                    if max_drawdown is not None
                    else None
                ),
                "Unit": "%",
            },

            {
                "Metric": "Sharpe ratio",
                "Value": sharpe,
                "Unit": "",
            },

            {
                "Metric": "Recent annualized volatility",
                "Value": (
                    float(recent_volatility) * 100
                    if recent_volatility is not None
                    else None
                ),
                "Unit": "%",
            },

            {
                "Metric": "Beta",
                "Value": beta,
                "Unit": "",
            },
        ]


        risk_df = pd.DataFrame(
            risk_rows
        )


        # Format values
        if not risk_df.empty:

            risk_df["Value"] = risk_df[
                "Value"
            ].apply(
                lambda x: (
                    "N/A"
                    if x is None
                    else (
                        "N/A"
                        if isinstance(x, float)
                        and pd.isna(x)
                        else f"{x:.4f}"
                    )
                )
            )


        display_dataframe(
            risk_df
        )


        st.caption(
            "Risk metrics are calculated from historical "
            "data. They describe past behavior and do not "
            "guarantee future risk or returns."
        )


    except Exception as risk_display_error:

        st.error(
            f"Risk analysis could not be displayed: "
            f"{risk_display_error}"
        )

        st.info(
            "The market-data portion of the application "
            "is still working. The risk calculation may "
            "need more historical data or a valid benchmark."
        )


# ============================================================
# TAB 5 — MACHINE LEARNING
# ============================================================

with tab5:

    st.subheader(
        "🤖 Machine-learning direction analysis"
    )

    st.caption(
        "The model uses historical technical features "
        "to classify the next observed direction as "
        "UP or DOWN. This is not a guaranteed forecast."
    )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Training the time-ordered model..."
        ):

            ml = run_direction_model(
                df
            )

    except Exception as ml_error:

        ml = {
            "available": False,
            "message": (
                f"Machine-learning model error: "
                f"{ml_error}"
            ),
        }


    # --------------------------------------------------------
    # CHECK RESULT
    # --------------------------------------------------------

    if not isinstance(
        ml,
        dict,
    ):

        st.error(
            "The ML model returned an invalid result."
        )

    elif ml.get("available"):

        try:

            # ------------------------------------------------
            # TOP ML METRICS
            # ------------------------------------------------

            m1, m2, m3 = st.columns(3)


            accuracy = ml.get(
                "accuracy"
            )

            f1_score = ml.get(
                "f1"
            )

            next_direction = ml.get(
                "next_direction",
                "N/A",
            )


            # Accuracy

            if accuracy is not None:

                accuracy_text = (
                    f"{float(accuracy) * 100:.2f}%"
                )

            else:

                accuracy_text = "N/A"


            # F1

            if f1_score is not None:

                f1_text = (
                    f"{float(f1_score) * 100:.2f}%"
                )

            else:

                f1_text = "N/A"


            m1.metric(
                "Historical test accuracy",
                accuracy_text,
            )


            m2.metric(
                "F1",
                f1_text,
            )


            m3.metric(
                "Latest direction class",
                str(next_direction),
            )


            # ------------------------------------------------
            # PRECISION
            # ------------------------------------------------

            precision = ml.get(
                "precision"
            )

            if precision is not None:

                st.write(
                    "Precision:",
                    f"{float(precision) * 100:.2f}%",
                )

            else:

                st.write(
                    "Precision:",
                    "N/A",
                )


            # ------------------------------------------------
            # RECALL
            # ------------------------------------------------

            recall = ml.get(
                "recall"
            )

            if recall is not None:

                st.write(
                    "Recall:",
                    f"{float(recall) * 100:.2f}%",
                )

            else:

                st.write(
                    "Recall:",
                    "N/A",
                )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = ml.get(
                "confidence"
            )

            if confidence is not None:

                st.write(
                    "Model probability:",
                    f"{float(confidence) * 100:.2f}%",
                )

            else:

                st.write(
                    "Model probability:",
                    "N/A",
                )


            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            st.subheader(
                "Confusion matrix"
            )


            confusion_matrix = ml.get(
                "confusion_matrix"
            )


            if confusion_matrix is not None:

                try:

                    confusion_df = pd.DataFrame(
                        confusion_matrix
                    )

                    # Handle normal 2x2 matrix
                    if (
                        confusion_df.shape[0] == 2
                        and confusion_df.shape[1] == 2
                    ):

                        confusion_df.index = [
                            "Actual DOWN",
                            "Actual UP",
                        ]

                        confusion_df.columns = [
                            "Predicted DOWN",
                            "Predicted UP",
                        ]

                    display_dataframe(
                        confusion_df
                    )

                except Exception as matrix_error:

                    st.warning(
                        "Confusion matrix could not "
                        f"be displayed: {matrix_error}"
                    )

            else:

                st.info(
                    "Confusion matrix was not returned "
                    "by the model."
                )


            # ------------------------------------------------
            # MODEL WARNING
            # ------------------------------------------------

            warning = ml.get(
                "warning"
            )

            if warning:

                st.warning(
                    str(warning)
                )


        except Exception as ml_display_error:

            st.error(
                "The ML model ran, but its results "
                "could not be displayed."
            )

            st.code(
                str(ml_display_error)
            )

    else:

        message = ml.get(
            "message",
            "The ML model could not be trained.",
        )

        st.warning(
            str(message)
        )

        st.info(
            "Try selecting a longer historical period "
            "such as 1y or 2y. The model needs enough "
            "historical rows after technical indicators "
            "and missing values are removed."
        )


# ============================================================
# TAB 6 — FUNDAMENTALS
# ============================================================

with tab6:

    st.subheader(
        "🏢 Selected fundamentals"
    )

    try:

        if not fundamentals:

            st.info(
                "Fundamental data was not available "
                "for this ticker."
            )

        else:

            # ----------------------------------------------
            # Convert dictionary into rows
            # ----------------------------------------------

            fundamental_rows = []


            for key, value in fundamentals.items():

                # Skip None
                if value is None:
                    continue


                # Skip NaN
                try:

                    if pd.isna(value):

                        continue

                except Exception:

                    pass


                # Make complex values safe
                if isinstance(
                    value,
                    (
                        dict,
                        list,
                        tuple,
                    ),
                ):

                    value = str(value)


                fundamental_rows.append(
                    {
                        "Metric": str(key),
                        "Value": value,
                    }
                )


            if fundamental_rows:

                fundamentals_df = pd.DataFrame(
                    fundamental_rows
                )


                # Convert all values safely to strings
                fundamentals_df[
                    "Metric"
                ] = fundamentals_df[
                    "Metric"
                ].astype(str)


                fundamentals_df[
                    "Value"
                ] = fundamentals_df[
                    "Value"
                ].apply(
                    lambda x: (
                        "N/A"
                        if x is None
                        else str(x)
                    )
                )


                display_dataframe(
                    fundamentals_df
                )

            else:

                st.info(
                    "No usable fundamental values "
                    "were returned."
                )


    except Exception as fundamental_display_error:

        st.error(
            "Fundamental data could not be displayed."
        )

        st.code(
            str(fundamental_display_error)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Observed market data and retrieved news are distinct "
    "from calculated indicators and machine-learning outputs. "
    "Verify important information with primary sources."
)