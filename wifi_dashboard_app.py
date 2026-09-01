import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import re
import html
import io
from collections import Counter


st.set_page_config(
    page_title="Wi-Fi Experience Dashboard",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)


PRIMARY = "#0B5FFF"
SECONDARY = "#0A2540"
ACCENT = "#00A3A3"
BG = "#F5F8FC"
CARD = "#FFFFFF"
MUTED = "#5B6B82"


st.markdown(
    f"""
    <style>
            a {{
                text-decoration: none;
                color: #464feb;
            }}
      .stApp {{
        background:
          radial-gradient(circle at 15% 20%, rgba(11,95,255,0.08), transparent 30%),
          radial-gradient(circle at 85% 0%, rgba(0,163,163,0.08), transparent 28%),
          {BG};
      }}
      .block-container {{
        padding-top: 1.1rem;
        padding-bottom: 1.8rem;
      }}
      .hero {{
        background: linear-gradient(120deg, {SECONDARY}, {PRIMARY});
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(10,37,64,0.18);
      }}
      .hero h1 {{
        font-size: 1.7rem;
        margin: 0;
      }}
      .hero p {{
        margin: 0.25rem 0 0 0;
        color: #DFE8FF;
      }}
      .metric-card {{
        background: {CARD};
        border: 1px solid #E6ECF5;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 2px 8px rgba(19,31,55,0.05);
      }}
      .metric-label {{
        color: {MUTED};
        font-size: 0.82rem;
        margin-bottom: 0.2rem;
      }}
      .metric-value {{
        color: {SECONDARY};
        font-weight: 700;
        font-size: 1.35rem;
        line-height: 1.25;
      }}
            .key-findings-list {{
                margin: 0.35rem 0 0 1rem;
                color: {SECONDARY};
                font-size: 0.92rem;
                line-height: 1.4;
            }}
            .key-findings-list li {{
                margin-bottom: 0.28rem;
            }}
            .caption-note {{
                color: #6b7280;
                font-size: 0.82rem;
                line-height: 1.4;
                margin-top: 0.2rem;
                white-space: pre-line;
            }}
      .section-title {{
        margin-top: 0.3rem;
        margin-bottom: 0.6rem;
        color: {SECONDARY};
        font-weight: 700;
      }}
            .stTable table {{
                border-collapse: collapse;
            }}
            .stTable tr th, .stTable tr td {{
                border: 1px solid #e6e6e6;
            }}
            .stTable tr th {{
                background-color: #f5f5f5;
            }}
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_CSV = "nps_wifi_score_all2026.csv"
# Pre-filtered to wifi_comment == 1 rows only, avoids loading the full 1.4M-row raw file on every run.
DEFAULT_SENTIMENT_CSV = "nps_wifi_comment_only.csv"
# Row count of the full nps_comment_all.csv (all comments, not just WiFi-related) used for the "Total Comments" metric.
TOTAL_COMMENT_ROWS_DEFAULT = 1430569

TEXT_COL = "LIKELIHOOD_RECOMMEND_ORIG_LANG_TXT"
WIFI_CATEGORY_COL = "Wifif Category"

SCORE_COLS = [
    "WIFI_SATISFACTION_SCR",
    "WIFI_SPEED_SCR",
    "WIFI_PRICE_SCR",
    "WIFI_RLBLTY_SCR",
    "WIFI_CNCT_EFFRT_SCR",
]

SCORE_LABELS = {
    "WIFI_SATISFACTION_SCR": "WiFi Satisfaction",
    "WIFI_SPEED_SCR": "Speed",
    "WIFI_PRICE_SCR": "Price",
    "WIFI_RLBLTY_SCR": "Reliability",
    "WIFI_CNCT_EFFRT_SCR": "Connection Effort",
}

CATEGORY_COLS = [
    "NET_PROMO_CATG",
    "FLEET_CD",
    "SUBFLEET_CD",
    "CABIN_FLOWN_CD",
    "SEG_DEP_AIRPRT_IATA_CD",
    "SEG_ARVL_AIRPRT_IATA_CD",
]


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_sentiment_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data(show_spinner=False)
def load_uploaded_data(upload_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(upload_bytes))


@st.cache_data(show_spinner=False)
def load_uploaded_sentiment_data(upload_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(upload_bytes))


def to_numeric_if_exists(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def detect_wifi_scale(df: pd.DataFrame) -> tuple[str, float, float]:
    if "WIFI_SATISFACTION_SCR" not in df.columns or df["WIFI_SATISFACTION_SCR"].dropna().empty:
        return "unknown", np.nan, np.nan

    p99 = df["WIFI_SATISFACTION_SCR"].quantile(0.99)
    if p99 <= 5.5:
        return "1-5", 2, 4
    return "0-100", 40, 80


def format_number(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class='metric-card'>
          <div class='metric-label'>{label}</div>
          <div class='metric-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_caption(text: str):
    st.markdown(
        f"<div class='caption-note'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def style_table(df_table: pd.DataFrame, precision: int = 2):
    return (
        df_table.style
        .hide(axis="index")
        .format(precision=precision)
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#f5f5f5"), ("border", "1px solid #e6e6e6")]},
                {"selector": "td", "props": [("border", "1px solid #e6e6e6")]},
            ]
        )
    )


def wifi_satisfaction_label(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(index=series.index, dtype="object")

    max_val = series.quantile(0.99)
    if max_val <= 5.5:
        label_map = {
            1: "Extremely poor",
            2: "Poor",
            3: "Average",
            4: "Good",
            5: "Excellent",
        }
        return series.round().map(label_map).mask(series == -1, "Not applicable")

    anchors = [1.0, 25.75, 50.5, 75.25, 100.0]
    labels = [
        "Extremely poor",
        "Poor",
        "Average",
        "Good",
        "Excellent",
    ]

    def nearest_label(v):
        if pd.isna(v):
            return np.nan
        if v == -1:
            return "Not applicable"
        idx = int(np.argmin([abs(v - a) for a in anchors]))
        return labels[idx]

    return series.apply(nearest_label)


st.markdown(
    """
    <div class='hero'>
      <h1>Wi-Fi Experience Analytics Dashboard</h1>
      <p>Interactive exploration of Wi-Fi satisfaction patterns by customer sentiment and flight attributes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Data + Filters")

applied_state = st.session_state.get("applied_dashboard_state")

if applied_state and applied_state.get("main_upload_bytes"):
    controls_source_df = load_uploaded_data(applied_state["main_upload_bytes"])
else:
    try:
        controls_source_df = load_data(DEFAULT_CSV)
    except Exception as exc:
        st.error(f"Could not load default file {DEFAULT_CSV}: {exc}")
        st.stop()

if "SEG_DEP_DT" in controls_source_df.columns:
    controls_source_df["SEG_DEP_DT"] = pd.to_datetime(controls_source_df["SEG_DEP_DT"], errors="coerce")

controls_filter_cols = [c for c in ["NET_PROMO_CATG", "FLEET_CD", "SUBFLEET_CD", "CABIN_FLOWN_CD"] if c in controls_source_df.columns]
controls_options = {c: sorted(controls_source_df[c].dropna().astype(str).unique().tolist()) for c in controls_filter_cols}

default_date_range = None
if "SEG_DEP_DT" in controls_source_df.columns and controls_source_df["SEG_DEP_DT"].notna().any():
    default_date_range = (
        controls_source_df["SEG_DEP_DT"].min().date(),
        controls_source_df["SEG_DEP_DT"].max().date(),
    )

with st.sidebar:
    with st.form("dashboard_submit_form"):
        st.caption("Defaults are auto-loaded. Upload only if you want to override.")
        main_upload = st.file_uploader("Upload survey CSV (optional)", type=["csv"], key="main_upload_form")
        include_sentiment_form = st.checkbox(
            "Include qualitative data",
            value=applied_state["include_sentiment"] if applied_state else True,
        )
        sentiment_upload = st.file_uploader(
            "Upload qualitative CSV (optional)",
            type=["csv"],
            key="sentiment_upload_form",
            disabled=not include_sentiment_form,
        )

        if default_date_range is not None:
            applied_date_range = applied_state["date_range"] if applied_state else default_date_range
            if (
                applied_date_range is None
                or len(applied_date_range) != 2
                or applied_date_range[0] is None
                or applied_date_range[1] is None
            ):
                applied_date_range = default_date_range

            start_default, end_default = applied_date_range
            start_col, end_col = st.columns(2)
            with start_col:
                start_date_input = st.date_input(
                    "Start date",
                    value=start_default,
                    min_value=default_date_range[0],
                    max_value=default_date_range[1],
                )
            with end_col:
                end_date_input = st.date_input(
                    "End date",
                    value=end_default,
                    min_value=default_date_range[0],
                    max_value=default_date_range[1],
                )
            date_range_input = (start_date_input, end_date_input)
        else:
            date_range_input = None

        selections_input = {}
        applied_selections = applied_state["selections"] if applied_state else {}
        for c in controls_filter_cols:
            options = controls_options[c]
            prior_default = [v for v in applied_selections.get(c, options) if v in options]
            if not prior_default:
                prior_default = options
            selections_input[c] = st.multiselect(c, options, default=prior_default)

        submitted = st.form_submit_button("Submit", use_container_width=True)

if submitted:
    final_date_range = date_range_input
    if final_date_range is not None and len(final_date_range) == 2 and final_date_range[0] > final_date_range[1]:
        final_date_range = (final_date_range[1], final_date_range[0])

    st.session_state["applied_dashboard_state"] = {
        "main_upload_bytes": main_upload.getvalue() if main_upload is not None else None,
        "include_sentiment": include_sentiment_form,
        "sentiment_upload_bytes": sentiment_upload.getvalue() if sentiment_upload is not None else None,
        "date_range": final_date_range,
        "selections": selections_input,
    }

if "applied_dashboard_state" not in st.session_state:
    initial_selections = {c: controls_options[c] for c in controls_filter_cols}
    st.session_state["applied_dashboard_state"] = {
        "main_upload_bytes": None,
        "include_sentiment": True,
        "sentiment_upload_bytes": None,
        "date_range": default_date_range,
        "selections": initial_selections,
    }

applied_state = st.session_state["applied_dashboard_state"]

if applied_state.get("main_upload_bytes"):
    try:
        df_raw = load_uploaded_data(applied_state["main_upload_bytes"])
    except Exception as exc:
        st.error(f"Could not read uploaded survey CSV: {exc}")
        st.stop()
else:
    try:
        df_raw = load_data(DEFAULT_CSV)
    except Exception as exc:
        st.error(f"Could not load default file {DEFAULT_CSV}: {exc}")
        st.stop()

df = df_raw.copy()
if "SEG_DEP_DT" in df.columns:
    df["SEG_DEP_DT"] = pd.to_datetime(df["SEG_DEP_DT"], errors="coerce")
    df["DEP_MONTH"] = df["SEG_DEP_DT"].dt.to_period("M").astype(str)

numeric_candidates = SCORE_COLS + ["LIKELIHOOD_RECOMMEND_SCL"]
df = to_numeric_if_exists(df, numeric_candidates)

for c in CATEGORY_COLS:
    if c in df.columns:
        df[c] = df[c].astype("string")

scale_label, low_thr, high_thr = detect_wifi_scale(df)

selections = applied_state.get("selections", {})
for col, vals in selections.items():
    if col in df.columns and vals:
        df = df[df[col].astype(str).isin(vals)]

date_range = applied_state.get("date_range")
if date_range is not None and len(date_range) == 2 and "SEG_DEP_DT" in df.columns:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["SEG_DEP_DT"] >= start_dt) & (df["SEG_DEP_DT"] <= end_dt)]

enable_sentiment = bool(applied_state.get("include_sentiment", True))
df_sentiment_raw = None
if enable_sentiment:
    if applied_state.get("sentiment_upload_bytes"):
        try:
            df_sentiment_raw = load_uploaded_sentiment_data(applied_state["sentiment_upload_bytes"])
        except Exception as exc:
            st.warning(f"Could not read uploaded qualitative CSV: {exc}")
    else:
        try:
            df_sentiment_raw = load_sentiment_data(DEFAULT_SENTIMENT_CSV)
        except Exception as exc:
            st.warning(f"Could not load default qualitative file {DEFAULT_SENTIMENT_CSV}: {exc}")

df_sentiment = None
total_comment_rows = 0
if enable_sentiment and df_sentiment_raw is not None and not df_sentiment_raw.empty:
    if applied_state.get("sentiment_upload_bytes"):
        # Uploaded file is assumed to be the raw, unfiltered export with a wifi_comment flag column.
        df_sentiment = df_sentiment_raw[df_sentiment_raw["wifi_comment"] == 1].copy()
        total_comment_rows = len(df_sentiment_raw)
    else:
        # Default file is already pre-filtered to WiFi comments only.
        df_sentiment = df_sentiment_raw.copy()
        total_comment_rows = TOTAL_COMMENT_ROWS_DEFAULT
    del df_sentiment_raw
    if "SEG_DEP_DT" in df_sentiment.columns:
        df_sentiment["SEG_DEP_DT"] = pd.to_datetime(df_sentiment["SEG_DEP_DT"], errors="coerce")

    if "SURVEY_RESPNS_ID" in df_sentiment.columns:
        df_sentiment["SURVEY_RESPNS_ID"] = df_sentiment["SURVEY_RESPNS_ID"].astype(str)

    for c in controls_filter_cols:
        if c in df_sentiment.columns:
            df_sentiment[c] = df_sentiment[c].astype("string")
            vals = selections.get(c, [])
            if vals:
                df_sentiment = df_sentiment[df_sentiment[c].astype(str).isin(vals)]

    if date_range is not None and len(date_range) == 2 and "SEG_DEP_DT" in df_sentiment.columns:
        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_sentiment = df_sentiment[(df_sentiment["SEG_DEP_DT"] >= start_dt) & (df_sentiment["SEG_DEP_DT"] <= end_dt)]

    # if "SURVEY_RESPNS_ID" in df.columns and "SURVEY_RESPNS_ID" in df_sentiment.columns:
    #     main_ids = set(df["SURVEY_RESPNS_ID"].astype(str).dropna().unique().tolist())
    #     df_sentiment = df_sentiment[df_sentiment["SURVEY_RESPNS_ID"].isin(main_ids)]


st.markdown("### Coverage & Survey Bias")


st.markdown("#### Basic Summary")

k_total = len(df)
k_wifi_total = len(df[~df['WIFI_SATISFACTION_SCR'].isnull()])
k_avg_ltr = df["LIKELIHOOD_RECOMMEND_SCL"].mean() if "LIKELIHOOD_RECOMMEND_SCL" in df.columns else np.nan
k_dt_min = df["SEG_DEP_DT"].min().date() if "SEG_DEP_DT" in df.columns and df["SEG_DEP_DT"].notna().any() else "NA"
k_dt_max = df["SEG_DEP_DT"].max().date() if "SEG_DEP_DT" in df.columns and df["SEG_DEP_DT"].notna().any() else "NA"

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("Min Date", str(k_dt_min))
with m2:
    metric_card("Max Date", str(k_dt_max))
with m3:
    metric_card("Total Survey Responses", format_number(k_total))
with m4:
    # metric_card("Avg Likelihood Recommended", format_number(k_avg_ltr))
    metric_card("Total WiFi Satisfaction Responses", format_number(k_wifi_total))
with m5:
    metric_card("% WiFi Satisfaction Responses", format_number(100*k_wifi_total/k_total))

st.markdown("")
c1,c2 = st.columns(2)

nps_response_rate = (
    df.groupby("NET_PROMO_CATG", dropna=False)
    .agg(
        Total_Responses=("NET_PROMO_CATG", "size"),
        WiFi_Response_Count=("WIFI_SATISFACTION_SCR", "count"),
    )
    .reset_index()
)
nps_response_rate["NPS Category"] = nps_response_rate["NET_PROMO_CATG"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
nps_response_rate["WiFi Response Rate %"] = (
    nps_response_rate["WiFi_Response_Count"].div(nps_response_rate["Total_Responses"].where(nps_response_rate["Total_Responses"] > 0, np.nan))
    * 100
)
nps_response_rate = nps_response_rate[["NPS Category", "Total_Responses", "WiFi_Response_Count", "WiFi Response Rate %"]].sort_values("WiFi Response Rate %", ascending=False)

with c1:
    st.markdown("#### WiFi Response Rate by NPS Category")
    st.table(style_table(nps_response_rate.round(2), precision=2))
    render_caption( "This table shows how frequently customers in each NPS category choose to answer the WiFi satisfaction question.  \n"
        "Higher response rates mean WiFi feedback is available for more of that customer group.  \n"
        "Large differences across NPS groups indicate response bias, meaning some customer groups are more likely than others to share WiFi feedback.")

insights = []
    
wifi_response_rate = (100 * k_wifi_total / k_total) if k_total > 0 else np.nan
if not pd.isna(wifi_response_rate):
    insights.append(f"WiFi response rate is {wifi_response_rate:.2f}% ({int(k_wifi_total):,} of {int(k_total):,} responses).")

if not nps_response_rate.empty:
    top_rate_row = nps_response_rate.sort_values("WiFi Response Rate %", ascending=False).iloc[0]
    insights.append(
        f"Highest WiFi response rate is in {top_rate_row['NPS Category']} at {top_rate_row['WiFi Response Rate %']:.2f}% "
        f"({int(top_rate_row['WiFi_Response_Count']):,}/{int(top_rate_row['Total_Responses']):,})."
    )

if {"WIFI_SATISFACTION_SCR", "LIKELIHOOD_RECOMMEND_SCL"}.issubset(df.columns):
    nps_driver_df = df[["WIFI_SATISFACTION_SCR", "LIKELIHOOD_RECOMMEND_SCL"]].dropna()
    if not nps_driver_df.empty:
        nps_scale_top = 75.25 if (nps_driver_df["WIFI_SATISFACTION_SCR"].quantile(0.99) > 5.5) else 4
        nps_scale_low = 25.75 if nps_scale_top == 75.25 else 2
        low_nps = nps_driver_df.loc[nps_driver_df["WIFI_SATISFACTION_SCR"] <= nps_scale_low, "LIKELIHOOD_RECOMMEND_SCL"].mean()
        high_nps = nps_driver_df.loc[nps_driver_df["WIFI_SATISFACTION_SCR"] >= nps_scale_top, "LIKELIHOOD_RECOMMEND_SCL"].mean()
        if not pd.isna(low_nps) and not pd.isna(high_nps):
            insights.append(
                f"Avg NPS among high WiFi satisfaction customers is {high_nps:.2f} vs {low_nps:.2f} for low satisfaction "
                f"(gap {high_nps - low_nps:.2f})."
            )

if "DEP_MONTH" in df.columns and "WIFI_SATISFACTION_SCR" in df.columns:
    monthly_wifi_rate = (
        df.groupby("DEP_MONTH", dropna=False)["WIFI_SATISFACTION_SCR"]
        .apply(lambda s: s.notna().mean() * 100)
        .reset_index(name="wifi_response_rate_pct")
        .sort_values("DEP_MONTH")
    )
    if len(monthly_wifi_rate) >= 2:
        latest = monthly_wifi_rate.iloc[-1]
        previous = monthly_wifi_rate.iloc[-2]
        delta = latest["wifi_response_rate_pct"] - previous["wifi_response_rate_pct"]
        insights.append(
            f"Latest monthly WiFi response rate is {latest['wifi_response_rate_pct']:.2f}% in {latest['DEP_MONTH']}, "
            f"{delta:+.2f} pts vs prior month."
        )

if df_sentiment is not None and len(df) > 0:
    insights.append(f"Qualitative comments selected: {len(df_sentiment):,} ({(len(df_sentiment) / len(df)) * 100:.2f}% of filtered survey records).")

c2.markdown(" ")
c2.markdown(" ")
if insights:
    bullet_items = "".join([f"<li>{html.escape(item)}</li>" for item in insights[:5]])
    c2.markdown(
        f"""
        <div class='metric-card'>
          <div class='metric-label'>Key Findings</div>
          <ul class='key-findings-list'>
            {bullet_items}
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    c2.info("No insights available for current filters.")

st.markdown("---") 
st.markdown("### Relationship Between WiFi and NPS")


c1,c2 = st.columns(2)

if {"WIFI_SATISFACTION_SCR", "NET_PROMO_CATG"}.issubset(df.columns):
    rel_df = df[["WIFI_SATISFACTION_SCR", "NET_PROMO_CATG", "LIKELIHOOD_RECOMMEND_SCL"]].copy()
    rel_df["NET_PROMO_CATG"] = rel_df["NET_PROMO_CATG"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    rel_df["WiFi_Satisfaction_Level"] = wifi_satisfaction_label(rel_df["WIFI_SATISFACTION_SCR"])

    nps_relation = (
        rel_df.groupby("WiFi_Satisfaction_Level", dropna=False)
        .agg(
            Response_Count=("WIFI_SATISFACTION_SCR", "count"),
            Avg_NPS=("LIKELIHOOD_RECOMMEND_SCL", "mean"),
            Promoter_Pct=("NET_PROMO_CATG", lambda s: (s == "PROMOTER").mean() * 100),
            Passive_Pct=("NET_PROMO_CATG", lambda s: (s == "PASSIVE").mean() * 100),
            Detractor_Pct=("NET_PROMO_CATG", lambda s: (s == "DETRACTOR").mean() * 100),
        )
        .reset_index()
    )

    level_order = [
        "Not applicable",
        "Extremely poor",
        "Poor",
        "Average",
        "Good",
        "Excellent",
    ]
    nps_relation["WiFi_Satisfaction_Level"] = pd.Categorical(
        nps_relation["WiFi_Satisfaction_Level"],
        categories=level_order,
        ordered=True,
    )
    nps_relation = nps_relation.sort_values("WiFi_Satisfaction_Level").reset_index(drop=True)

    display_cols = [
        "WiFi_Satisfaction_Level",
        "Response_Count",
        "Avg_NPS",
        "Promoter_Pct",
        "Passive_Pct",
        "Detractor_Pct",
    ]
    with c1:
        st.markdown("")
        st.markdown("""
        Survey Rating Scale - 
        Extremely poor (1) |
        Poor (25.75) |
        Average (50.5) |
        Good (75.25) |
        Excellent (100) }
        Not applicable (-1) 
        """)
        
        st.dataframe(nps_relation[display_cols].round(2), use_container_width=True, hide_index=True)
        render_caption(
        "This table compares overall recommendation scores and NPS composition across different WiFi satisfaction levels.\n"
        "It helps determine whether customers who report better WiFi experiences are also more likely to become promoters and less likely to become detractors.\n"
        "WiFi Satisfaction Level = NaN refers to customers who did not provide a WiFi satisfaction rating but did submit an NPS rating.")

    dist_df = nps_relation.melt(
        id_vars="WiFi_Satisfaction_Level",
        value_vars=["Promoter_Pct", "Passive_Pct", "Detractor_Pct"],
        var_name="NPS_Type",
        value_name="Percent",
    )
    response_count_map = nps_relation.set_index("WiFi_Satisfaction_Level")["Response_Count"].to_dict()
    dist_df["Response_Count"] = dist_df["WiFi_Satisfaction_Level"].map(response_count_map)
    dist_df["NPS_Count"] = (dist_df["Percent"] * dist_df["Response_Count"] / 100).round().astype("Int64")

    fig_nps_dist = px.bar(
        dist_df,
        x="WiFi_Satisfaction_Level",
        y="Percent",
        color="NPS_Type",
        barmode="stack",
        title="NPS Distribution by Wi-Fi Satisfaction Level",
        hover_data={
            "Percent": ":.2f",
            "Response_Count": ":,.0f",
            "NPS_Count": ":,.0f",
        },
    )
    fig_nps_dist.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="Wi-Fi Satisfaction Level",
        yaxis_title="Percent",
        legend_title_text="NPS Type",
    )
    with c2:
        st.plotly_chart(fig_nps_dist, use_container_width=True)
        render_caption("This chart shows the promoter, passive, and detractor mix within each WiFi satisfaction level.\n"
        "Moving from left to right, increasing promoter share and decreasing detractor share would indicate that customers who report better WiFi experiences also tend to view their overall journey more positively.")


else:
    st.info("Required columns for WiFi-NPS relationship analysis are not available.")

# st.markdown("#### NPS Comparison Within Same Flight")

# required_cols = [
#     "WIFI_SATISFACTION_SCR",
#     "LIKELIHOOD_RECOMMEND_SCL",
#     "NET_PROMO_CATG",
#     "SEG_DEP_DT",
#     "SEG_DEP_AIRPRT_IATA_CD",
#     "SEG_ARVL_AIRPRT_IATA_CD",
# ]

# missing_prereqs = [c for c in required_cols if c not in df.columns]
# if missing_prereqs:
#     st.info("Cannot run same-flight comparison. Missing: " + ", ".join(missing_prereqs) + ".")
# else:
#     # Step 1: Create WiFi groups (Top Box vs Bottom Box)
#     top_box_cutoff = 75.25 if (df["WIFI_SATISFACTION_SCR"].dropna().quantile(0.99) > 5.5) else 4
#     bottom_box_cutoff = 25.75 if top_box_cutoff == 75.25 else 2

#     flight_df = df[
#         [
#             'OPERAT_FLIGHT_NBR',
#             "SEG_DEP_DT",
#             "SEG_DEP_AIRPRT_IATA_CD",
#             "SEG_ARVL_AIRPRT_IATA_CD",
#             "WIFI_SATISFACTION_SCR",
#             "LIKELIHOOD_RECOMMEND_SCL",
#             "NET_PROMO_CATG",
#         ]
#     ].copy()
#     flight_df["SEG_DEP_DT"] = pd.to_datetime(flight_df["SEG_DEP_DT"], errors="coerce")
#     flight_df["FLIGHT_DATE_KEY"] = flight_df["SEG_DEP_DT"].dt.date
#     flight_df["NET_PROMO_CATG"] = flight_df["NET_PROMO_CATG"].fillna("UNKNOWN").astype(str).str.upper().str.strip()

#     flight_df = flight_df.dropna(
#         subset=[
#             'OPERAT_FLIGHT_NBR',
#             "FLIGHT_DATE_KEY",
#             "SEG_DEP_AIRPRT_IATA_CD",
#             "SEG_ARVL_AIRPRT_IATA_CD",
#             "WIFI_SATISFACTION_SCR",
#             "LIKELIHOOD_RECOMMEND_SCL",
#         ]
#     )

#     flight_df["WiFi_Group"] = np.select(
#         [
#             flight_df["WIFI_SATISFACTION_SCR"] >= top_box_cutoff,
#             flight_df["WIFI_SATISFACTION_SCR"] <= bottom_box_cutoff,
#         ],
#         ["Good WiFi", "Bad WiFi"],
#         default="Other",
#     )
#     flight_df = flight_df[flight_df["WiFi_Group"].isin(["Good WiFi", "Bad WiFi"])]

#     # Step 2: Build a unique flight ID
#     flight_df["UNIQUE_FLIGHT_ID"] = (
#         flight_df['OPERAT_FLIGHT_NBR'].astype(str).str.strip()
#         + " | "
#         + flight_df["FLIGHT_DATE_KEY"].astype(str)
#         + " | "
#         + flight_df["SEG_DEP_AIRPRT_IATA_CD"].astype(str).str.strip()
#         + "-"
#         + flight_df["SEG_ARVL_AIRPRT_IATA_CD"].astype(str).str.strip()
#     )

#     # Step 3: Keep only flights where both groups exist and total responses > 5
#     eligible_flights = (
#         flight_df.groupby("UNIQUE_FLIGHT_ID")
#         .agg(
#             Group_Count=("WiFi_Group", "nunique"),
#             Total_Responses=("WiFi_Group", "size"),
#         )
#         .reset_index()
#     )
#     # eligible_ids = eligible_flights.loc[
#     #     (eligible_flights["Group_Count"] == 2) & (eligible_flights["Total_Responses"] >= 10),
#     #     "UNIQUE_FLIGHT_ID",
#     # ].tolist()

#     group_counts = (
#         flight_df.groupby(["UNIQUE_FLIGHT_ID", "WiFi_Group"])
#         .size()
#         .unstack(fill_value=0)
#     )

#     eligible_ids = group_counts[group_counts.min(axis=1) >= 5].index.tolist()
#     print(f"Eligible flights: {len(eligible_flights):,}")


#     if not eligible_ids:
#         st.info("No flights found with both Good WiFi and Bad WiFi groups and at least 5 total responses.")
#     else:
#         eligible_detail_df = (
#             flight_df[
#                 [
#                     "UNIQUE_FLIGHT_ID",
#                     "FLIGHT_DATE_KEY",
#                     'OPERAT_FLIGHT_NBR',
#                     "SEG_DEP_AIRPRT_IATA_CD",
#                     "SEG_ARVL_AIRPRT_IATA_CD",
#                 ]
#             ]
#             .drop_duplicates()
#             .copy()
#         )
#         eligible_detail_df = eligible_detail_df[eligible_detail_df["UNIQUE_FLIGHT_ID"].isin(eligible_ids)]
#         eligible_detail_df["FLIGHT_DATE_KEY"] = eligible_detail_df["FLIGHT_DATE_KEY"].astype(str)
#         eligible_detail_df["OPERAT_FLIGHT_NBR"] = eligible_detail_df["OPERAT_FLIGHT_NBR"].astype(str)
#         eligible_detail_df["ROUTE"] = (
#             eligible_detail_df["SEG_DEP_AIRPRT_IATA_CD"].astype(str).str.strip()
#             + "-"
#             + eligible_detail_df["SEG_ARVL_AIRPRT_IATA_CD"].astype(str).str.strip()
#         )

#         st.markdown("Select eligible flights to compare")
#         f1, f2, f3 = st.columns([1, 1.4, 0.8])
#         date_values = sorted(eligible_detail_df["FLIGHT_DATE_KEY"].unique().tolist())
#         selected_date = f1.selectbox(
#             "Date",
#             options=["Select date"] + date_values,
#             index=0,
#             key="sf_date",
#         )

#         date_filtered_df = eligible_detail_df.copy()
#         if selected_date != "Select date":
#             date_filtered_df = date_filtered_df[date_filtered_df["FLIGHT_DATE_KEY"] == selected_date]

#         date_filtered_df["FLIGHT_ROUTE_KEY"] = (
#             date_filtered_df["OPERAT_FLIGHT_NBR"]
#             + " | "
#             + date_filtered_df["ROUTE"]
#         )
#         combined_options = sorted(date_filtered_df["FLIGHT_ROUTE_KEY"].dropna().unique().tolist())
#         selected_flight_route = f2.selectbox(
#             "Flight Number + Airport Codes",
#             options=["Select flight+route"] + combined_options,
#             index=0,
#             key="sf_flight_route",
#             disabled=(selected_date == "Select date"),
#         )

#         show_flight_data = f3.button("Show flight data", key="show_flight_data_btn", use_container_width=True)

#         if show_flight_data:
#             if selected_date == "Select date":
#                 st.warning("Select Date first.")
#             elif selected_flight_route == "Select flight+route":
#                 st.warning("Select Flight Number + Airport Codes next.")
#             else:
#                 selected_ids = (
#                     date_filtered_df.loc[
#                         date_filtered_df["FLIGHT_ROUTE_KEY"] == selected_flight_route,
#                         "UNIQUE_FLIGHT_ID",
#                     ]
#                     .drop_duplicates()
#                     .tolist()
#                 )
#                 st.session_state["same_flight_submitted_ids"] = selected_ids
#                 st.session_state["same_flight_submitted_key"] = selected_flight_route

#         submitted_flight_ids = st.session_state.get("same_flight_submitted_ids", [])
#         submitted_key = st.session_state.get("same_flight_submitted_key", "")

#         if submitted_flight_ids:
#             compare_df = flight_df[flight_df["UNIQUE_FLIGHT_ID"].isin(submitted_flight_ids)].copy()

#             if compare_df.empty:
#                 st.info("Selected flight data is not available in the current filter context. Click Show flight data again.")
#             else:
#                 bad_group = compare_df[compare_df["WiFi_Group"] == "Bad WiFi"]
#                 good_group = compare_df[compare_df["WiFi_Group"] == "Good WiFi"]

#                 bad_responses = len(bad_group)
#                 good_responses = len(good_group)
#                 bad_avg_nps = bad_group["LIKELIHOOD_RECOMMEND_SCL"].mean()
#                 good_avg_nps = good_group["LIKELIHOOD_RECOMMEND_SCL"].mean()
#                 bad_promoter_pct = (bad_group["NET_PROMO_CATG"] == "PROMOTER").mean() * 100 if bad_responses > 0 else np.nan
#                 good_promoter_pct = (good_group["NET_PROMO_CATG"] == "PROMOTER").mean() * 100 if good_responses > 0 else np.nan
#                 bad_detractor_pct = (bad_group["NET_PROMO_CATG"] == "DETRACTOR").mean() * 100 if bad_responses > 0 else np.nan
#                 good_detractor_pct = (good_group["NET_PROMO_CATG"] == "DETRACTOR").mean() * 100 if good_responses > 0 else np.nan

#                 comparison_table = pd.DataFrame(
#                     {
#                         "Metric": ["Responses", "Avg NPS", "Promoter %", "Detractor %"],
#                         "Bad WiFi": [bad_responses, bad_avg_nps, bad_promoter_pct, bad_detractor_pct],
#                         "Good WiFi": [good_responses, good_avg_nps, good_promoter_pct, good_detractor_pct],
#                     }
#                 )
#                 comparison_table["Gap"] = comparison_table["Good WiFi"] - comparison_table["Bad WiFi"]

#                 st.caption(f"Comparison based on selected option: {selected_date} | {submitted_key}.")
#                 st.table(style_table(comparison_table.round(2), precision=2))
                

st.markdown("---") 
st.markdown("### Wi-Fi Satisfaction Baseline")

available_score_cols = [c for c in SCORE_COLS if c in df.columns]

if available_score_cols:
    top_box_cutoff = 75.25 if (df[available_score_cols].stack().quantile(0.99) > 5.5) else 4
    bottom_box_cutoff = 25.75 if top_box_cutoff == 75.25 else 2

    non_null_counts = df[available_score_cols].notna().sum()
    top_box_pct = (
        (df[available_score_cols].ge(top_box_cutoff) & df[available_score_cols].notna()).sum()
        .div(non_null_counts.where(non_null_counts > 0, np.nan))
        * 100
    )
    bottom_box_pct = (
        (df[available_score_cols].le(bottom_box_cutoff) & df[available_score_cols].notna()).sum()
        .div(non_null_counts.where(non_null_counts > 0, np.nan))
        * 100
    )

    summary = pd.DataFrame({
        "Response_Count": df[available_score_cols].count(),
        "Response_Rate_%": (df[available_score_cols].count() / len(df)) * 100,
        "Average_Score": df[available_score_cols].mean(),
        "Median": df[available_score_cols].median(),
        "Top_Box_%": top_box_pct,
        "Bottom_Box_%": bottom_box_pct,
        "NA_%": (df[available_score_cols].isna()).mean() * 100
    }).round(2)
    st.table(style_table(summary, precision=2))
    render_caption(
        "This table provides a baseline view of all WiFi-related survey attributes.\n"
        "Response Count = Number of customers answered each question.\n"
        "Response Rate % = Percentage of survey participants who provided a response for that metric.\n"
        "Top Box % = Customers reporting the most positive experience.\n"
        "Bottom Box % = Customers reporting the most negative experience.\n"
        "NA % = Customers who did not answer the question. " 
        )
    
else:
    st.info("Select at least one score column to view summary table.")

# st.markdown("### Grouped Summary ")

# available_group_cols = [c for c in CATEGORY_COLS if c in df.columns]
# default_group_cols = [c for c in ["NET_PROMO_CATG"] if c in available_group_cols]

# c1,c2 = st.columns(2)

# selected_group_cols = c1.multiselect(
#     "Group by columns",
#     options=available_group_cols,
#     default=default_group_cols,
# )

# metric_for_group = c2.selectbox(
#     "Metric column",
#     options=available_score_cols,
#     index=available_score_cols.index("WIFI_SATISFACTION_SCR") if "WIFI_SATISFACTION_SCR" in available_score_cols else 0,
# )

# if selected_group_cols:
#     grouped = (
#         df.groupby(selected_group_cols, dropna=False)[metric_for_group]
#         .agg(["count", "mean", "median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
#         .reset_index()
#         .rename(
#             columns={
#                 "count": "Non_Null_Count",
#                 "mean": "Average",
#                 "median": "Median",
#                 "<lambda_0>": "25th_Percentile",
#                 "<lambda_1>": "75th_Percentile",
#             }
#         )
#         .sort_values("Non_Null_Count", ascending=False)
#     )
#     st.dataframe(grouped.round(2), use_container_width=True)
# else:
#     st.info("Select at least one grouping column.")

st.markdown("---") 
st.markdown("### Trend and Segment Analysis")

numeric_for_y = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
# all_for_x = [c for c in df.columns if c not in ["SURVEY_ID", "SURVEY_RESPNS_ID", "SURVEY_AA_CUST_UNIQUE_ID", "SURVEY_LYLTY_ACCT_ID"]]

st.markdown("#### Month-on-Month Trend Analysis")
trend_controls_col, trend_chart_col = st.columns([1, 2])

if "DEP_MONTH" in df.columns:
    trend_df = df.copy()
    trend_df["NET_PROMO_CATG"] = trend_df["NET_PROMO_CATG"].fillna("UNKNOWN").astype(str).str.upper().str.strip()

    monthly_agg = {
        "wifi_satisfaction_score": ("WIFI_SATISFACTION_SCR", "mean"),
        "promoter_pct": ("NET_PROMO_CATG", lambda s: (s == "PROMOTER").mean() * 100),
        "detractor_pct": ("NET_PROMO_CATG", lambda s: (s == "DETRACTOR").mean() * 100),
        "response_volume": ("WIFI_SATISFACTION_SCR", "size"),
        "wifi_response_count": ("WIFI_SATISFACTION_SCR", "count"),
    }
    if "LIKELIHOOD_RECOMMEND_SCL" in trend_df.columns:
        monthly_agg["avg_nps"] = ("LIKELIHOOD_RECOMMEND_SCL", "mean")

    monthly = trend_df.groupby("DEP_MONTH", dropna=False).agg(**monthly_agg).reset_index().sort_values("DEP_MONTH")
    monthly["wifi_response_rate_pct"] = (
        monthly["wifi_response_count"].div(monthly["response_volume"].where(monthly["response_volume"] > 0, np.nan))
        * 100
    )

    if df_sentiment is not None and "SEG_DEP_DT" in df_sentiment.columns:
        sentiment_monthly = df_sentiment.copy()
        sentiment_monthly["DEP_MONTH"] = sentiment_monthly["SEG_DEP_DT"].dt.to_period("M").astype(str)
        complaint_vol = sentiment_monthly.groupby("DEP_MONTH", dropna=False).size().reset_index(name="wifi_complaint_volume")
        monthly = monthly.merge(complaint_vol, on="DEP_MONTH", how="left")
        monthly["wifi_complaint_volume"] = monthly["wifi_complaint_volume"].fillna(0)

    trend_options = {
        "WiFi satisfaction score": "wifi_satisfaction_score",
        "Promoter %": "promoter_pct",
        "Detractor %": "detractor_pct",
        "Response volume": "response_volume",
        "WiFi response rate %": "wifi_response_rate_pct",
    }
    if "avg_nps" in monthly.columns:
        trend_options["Avg NPS"] = "avg_nps"
    if "wifi_complaint_volume" in monthly.columns:
        trend_options["WiFi complaint volume"] = "wifi_complaint_volume"

    default_trend_metrics = ["WiFi satisfaction score", "Promoter %"]
    available_tb_cols = [c for c in SCORE_COLS if c in df.columns]

    if "applied_trend_chart_state" not in st.session_state:
        st.session_state["applied_trend_chart_state"] = {
            "selected_trends": [m for m in default_trend_metrics if m in trend_options],
            "include_tb": True,
            "tb_metric": "WIFI_SATISFACTION_SCR" if "WIFI_SATISFACTION_SCR" in available_tb_cols else (available_tb_cols[0] if available_tb_cols else None),
        }

    applied_trend_state = st.session_state["applied_trend_chart_state"]
    trend_default_selection = [m for m in applied_trend_state.get("selected_trends", []) if m in trend_options]
    if not trend_default_selection:
        trend_default_selection = [m for m in default_trend_metrics if m in trend_options]

    with trend_controls_col:
        with st.form("trend_chart_form"):
            selected_trends_input = st.multiselect(
                "Trend metrics",
                options=list(trend_options.keys()),
                default=trend_default_selection,
            )
            include_tb_input = st.checkbox(
                "Include Top/Bottom Box %",
                value=bool(applied_trend_state.get("include_tb", True)),
            )
            tb_metric_input = None
            if include_tb_input and available_tb_cols:
                prior_tb_metric = applied_trend_state.get("tb_metric")
                tb_idx = available_tb_cols.index(prior_tb_metric) if prior_tb_metric in available_tb_cols else 0
                tb_metric_input = st.selectbox(
                    "Top/Bottom metric",
                    options=available_tb_cols,
                    index=tb_idx,
                )
            trend_submit = st.form_submit_button("Update trend chart", use_container_width=True)

    if trend_submit:
        st.session_state["applied_trend_chart_state"] = {
            "selected_trends": selected_trends_input,
            "include_tb": include_tb_input,
            "tb_metric": tb_metric_input,
        }

    applied_trend_state = st.session_state["applied_trend_chart_state"]
    selected_trends = [m for m in applied_trend_state.get("selected_trends", []) if m in trend_options]
    include_tb_trend = bool(applied_trend_state.get("include_tb", True))
    tb_metric = applied_trend_state.get("tb_metric") if include_tb_trend else None
    if tb_metric not in available_tb_cols:
        tb_metric = available_tb_cols[0] if available_tb_cols else None

    trend_frames = []
    if selected_trends:
        selected_metric_cols = [trend_options[m] for m in selected_trends]
        trend_long = monthly[["DEP_MONTH"] + selected_metric_cols].melt(
            id_vars="DEP_MONTH",
            var_name="Metric_Key",
            value_name="Metric_Value",
        )

        reverse_trend_options = {v: k for k, v in trend_options.items()}
        trend_long["Metric"] = trend_long["Metric_Key"].map(reverse_trend_options)
        trend_frames.append(trend_long[["DEP_MONTH", "Metric", "Metric_Value"]])

    if include_tb_trend and tb_metric is not None:
        tb_metric_label = SCORE_LABELS.get(tb_metric, tb_metric)
        tb_top_cutoff = 75.25 if (df[tb_metric].dropna().quantile(0.99) > 5.5) else 4
        tb_bottom_cutoff = 25.75 if tb_top_cutoff == 75.25 else 2

        tb_monthly = (
            df.groupby("DEP_MONTH", dropna=False)
            .agg(
                Top_Box_Pct=(tb_metric, lambda s: (s.ge(tb_top_cutoff) & s.notna()).mean() * 100),
                Bottom_Box_Pct=(tb_metric, lambda s: (s.le(tb_bottom_cutoff) & s.notna()).mean() * 100),
            )
            .reset_index()
            .sort_values("DEP_MONTH")
        )

        tb_long = tb_monthly.melt(
            id_vars=["DEP_MONTH"],
            value_vars=["Top_Box_Pct", "Bottom_Box_Pct"],
            var_name="Metric",
            value_name="Metric_Value",
        )
        tb_long["Metric"] = tb_long["Metric"].replace(
            {
                "Top_Box_Pct": f"Top Box % ({tb_metric_label})",
                "Bottom_Box_Pct": f"Bottom Box % ({tb_metric_label})",
            }
        )
        trend_frames.append(tb_long[["DEP_MONTH", "Metric", "Metric_Value"]])

    if trend_frames:
        trend_combined = pd.concat(trend_frames, ignore_index=True)
        fig_line = px.line(
            trend_combined,
            x="DEP_MONTH",
            y="Metric_Value",
            color="Metric",
            markers=True,
            title="Monthly Trend: Selected Metrics",
        )
        fig_line.update_traces(line=dict(width=3), marker=dict(size=7))
        fig_line.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=55, b=10),
            xaxis_title="Month",
            yaxis_title="Metric Value",
            legend_title_text="Metric",
        )
        with trend_chart_col:
            st.plotly_chart(fig_line, use_container_width=True)
            render_caption(
                "This chart tracks how WiFi and customer metrics change over time. \n"
                "Consistent patterns may indicate underlying improvements or recurring issues, while sudden shifts may correspond to operational changes, service disruptions, or customer experience initiatives.\n"
                "Top Box and Bottom Box metrics focus on customers at the extremes of the satisfaction scale."
            )


    else:
        with trend_chart_col:
            st.info("Select at least one trend metric or enable Top/Bottom Box %.")
else:
    with trend_chart_col:
        st.info("DEP_MONTH is not available for trend analysis.")

st.markdown("#### Segment Analysis")
segment_controls_col, segment_chart_col = st.columns([1, 2])

segment_submit = False
x_bar = None
y_bars = []
primary_metric = None
agg_bar = "mean"
top_n = 5
segment_data = None
ranked_data = None
y_label = ""

if CATEGORY_COLS and numeric_for_y:
    with segment_controls_col:
        if "applied_segment_chart_state" not in st.session_state:
            st.session_state["applied_segment_chart_state"] = {
                "x_bar": "FLEET_CD" if "FLEET_CD" in CATEGORY_COLS else CATEGORY_COLS[0],
                "y_bars": ["WIFI_SATISFACTION_SCR"] if "WIFI_SATISFACTION_SCR" in numeric_for_y else [numeric_for_y[0]],
                "agg_bar": "mean",
                "top_n": top_n,
            }

        applied_segment_state = st.session_state["applied_segment_chart_state"]

        with st.form("segment_chart_form"):
            x_default = applied_segment_state.get("x_bar")
            y_defaults = [m for m in applied_segment_state.get("y_bars", []) if m in numeric_for_y]
            if not y_defaults:
                y_defaults = ["WIFI_SATISFACTION_SCR"] if "WIFI_SATISFACTION_SCR" in numeric_for_y else [numeric_for_y[0]]
            agg_default = applied_segment_state.get("agg_bar", "mean")

            x_idx = CATEGORY_COLS.index(x_default) if x_default in CATEGORY_COLS else 0
            agg_options = ["mean", "median", "count", "sum"]
            agg_idx = agg_options.index(agg_default) if agg_default in agg_options else 0

            x_bar_input = st.selectbox("Segment dimension", options=CATEGORY_COLS, index=x_idx)
            y_bars_input = st.multiselect("Metrics", options=numeric_for_y, default=y_defaults)
            agg_bar_input = st.selectbox("Aggregation", options=agg_options, index=agg_idx)
            top_n_input = st.slider("Top segments shown", min_value=5, max_value=30, value=int(applied_segment_state.get("top_n", top_n)))
            segment_submit = st.form_submit_button("Update segment charts", use_container_width=True)

    if segment_submit:
        final_metrics = y_bars_input if y_bars_input else (["WIFI_SATISFACTION_SCR"] if "WIFI_SATISFACTION_SCR" in numeric_for_y else [numeric_for_y[0]])
        st.session_state["applied_segment_chart_state"] = {
            "x_bar": x_bar_input,
            "y_bars": final_metrics,
            "agg_bar": agg_bar_input,
            "top_n": top_n_input,
        }

    applied_segment_state = st.session_state["applied_segment_chart_state"]
    x_bar = applied_segment_state.get("x_bar")
    y_bars = [m for m in applied_segment_state.get("y_bars", []) if m in numeric_for_y]
    if not y_bars:
        y_bars = ["WIFI_SATISFACTION_SCR"] if "WIFI_SATISFACTION_SCR" in numeric_for_y else [numeric_for_y[0]]
    primary_metric = y_bars[0]
    agg_bar = applied_segment_state.get("agg_bar", "mean")
    top_n = int(applied_segment_state.get("top_n", 12))

if x_bar in df.columns and primary_metric in df.columns:
    if agg_bar == "count":
        segment_data = df.groupby(x_bar, dropna=False).size().reset_index(name="value")
        segment_data["response_count"] = segment_data["value"]
        y_label = "Count"
    else:
        segment_data = (
            df.groupby(x_bar, dropna=False)
            .agg(
                value=(primary_metric, agg_bar),
                response_count=(primary_metric, "count"),
            )
            .reset_index()
        )
        y_label = f"{agg_bar.title()} of {primary_metric}"

    segment_data = segment_data.sort_values("value", ascending=False)
    ranked_data = segment_data.head(top_n).copy()
    ranked_data[x_bar] = ranked_data[x_bar].astype(str)

    with segment_controls_col:
        top_segment_values = ", ".join(ranked_data[x_bar].dropna().astype(str).tolist())
        st.markdown(f"Top {top_n} {x_bar} values: {top_segment_values}")
    # with segment_controls_col:
    #     st.markdown(f"Top {top_n} {x_bar} values")
    #     preview_df = ranked_data[[x_bar, "value", "response_count"]].rename(
    #         columns={"value": y_label, "response_count": "Response Count"}
    #     )
    #     st.table(style_table(preview_df.round(2), precision=2))

    if ranked_data is not None and segment_data is not None:
        with segment_chart_col:
            tab1, tab2 = st.tabs(["Performance Matrix", "Ranked Segment Points"])

            with tab1:
                if agg_bar == "count":
                    st.info("Performance matrix is most useful for mean/median/sum. Use Ranked Segment Bars for count-based analysis.")
                    render_caption(
                        "This represents a relationship between segment score and response volume.\n"
                        "To read, switch aggregation from count to mean/median/sum to see actionable quadrants."
                    )
                else:
                    matrix_df = segment_data.copy()
                    matrix_df[x_bar] = matrix_df[x_bar].astype(str)
                    fig_matrix = px.scatter(
                        matrix_df,
                        x="response_count",
                        y="value",
                        size="response_count",
                        color="value",
                        color_continuous_scale=["#A8DADC", "#0B5FFF"],
                        hover_name=x_bar,
                        title=f"Segment Performance Matrix: {y_label}",
                    )
                    fig_matrix.add_hline(
                        y=matrix_df["value"].mean(),
                        line_dash="dot",
                        line_color="#0A2540",
                        annotation_text="Overall avg",
                        annotation_position="top right",
                    )
                    fig_matrix.add_vline(
                        x=matrix_df["response_count"].median(),
                        line_dash="dot",
                        line_color="#00A3A3",
                        annotation_text="Median volume",
                        annotation_position="top right",
                    )
                    fig_matrix.update_layout(
                        height=430,
                        margin=dict(l=10, r=10, t=55, b=10),
                        xaxis_title="Response volume",
                        yaxis_title=y_label,
                    )
                    st.plotly_chart(fig_matrix, use_container_width=True)
                    render_caption(
                        "In the above graph, each point is one segment with its performance on the Y-axis and response volume on the X-axis; larger bubbles mean more responses. Darker/stronger color = higher value based on the selected scale. \n"
                        "Top-right: high score + high volume. Strong segments at scale.\n"
                        "Bottom-right: low score + high volume. Biggest improvement opportunities.\n"
                        "Top-left: high score + low volume. Niche strengths.\n"
                        "Bottom-left: low score + low volume. Lower priority issues.\n"
                        "Horizontal dotted line = overall average metric value.\n"
                        "Vertical dotted line = median response volume.\n"
                    )

            with tab2:
                ranked_order = ranked_data[x_bar].tolist()

                if agg_bar == "count":
                    point_data = ranked_data.copy()
                    fig_points = px.scatter(
                        point_data,
                        x=x_bar,
                        y="value",
                        title=f"Top {top_n} Segments: Count",
                        hover_data={"response_count": ":,.0f"},
                    )
                    fig_points.update_traces(marker=dict(size=10, color=PRIMARY))
                    fig_points.update_layout(
                        height=430,
                        margin=dict(l=10, r=10, t=55, b=10),
                        xaxis_title=x_bar,
                        yaxis_title="Count",
                        showlegend=False,
                    )
                else:
                    point_parts = []
                    for metric_col in y_bars:
                        metric_part = (
                            df.groupby(x_bar, dropna=False)
                            .agg(value=(metric_col, agg_bar), response_count=(metric_col, "count"))
                            .reset_index()
                        )
                        metric_part[x_bar] = metric_part[x_bar].astype(str)
                        metric_part["Metric"] = SCORE_LABELS.get(metric_col, metric_col)
                        point_parts.append(metric_part)

                    point_data = pd.concat(point_parts, ignore_index=True)
                    point_data = point_data[point_data[x_bar].isin(ranked_order)]

                    fig_points = px.scatter(
                        point_data,
                        x=x_bar,
                        y="value",
                        color="Metric",
                        title=f"Top {top_n} Segments: {agg_bar.title()} across selected metrics",
                        hover_data={"response_count": ":,.0f"},
                    )
                    fig_points.update_traces(marker=dict(size=9))
                    fig_points.update_layout(
                        height=430,
                        margin=dict(l=10, r=10, t=55, b=10),
                        xaxis_title=x_bar,
                        yaxis_title=f"{agg_bar.title()} value",
                        legend_title_text="Metric",
                    )

                fig_points.update_xaxes(
                    type="category",
                    categoryorder="array",
                    categoryarray=ranked_order,
                    tickmode="array",
                    tickvals=ranked_order,
                    ticktext=ranked_order,
                    tickangle=-45,
                    automargin=True,
                )
                fig_points.update_layout(
                    height=430,
                    margin=dict(l=10, r=10, t=55, b=10),
                    xaxis_title=x_bar,
                )
                st.plotly_chart(fig_points, use_container_width=True)
                render_caption(
                    "This point chart compares selected metrics across the same Top-N categorical segments.\n"
                    "Different point colors represent different selected metrics, and the Y-axis shows the chosen aggregation value.\n"
                    "Segment order is based on the primary metric (first selected metric)."
                )
    else:
        with segment_chart_col:
            st.info("Segment analysis requires at least one categorical dimension and one numeric metric.")


    st.markdown("### Difference Between Highest and Lowest Satisfaction Groups")


if "LIKELIHOOD_RECOMMEND_SCL" in df.columns:
    gap_rows = []
    metric_map = {
        "WiFi Satisfaction" : "WIFI_SATISFACTION_SCR",
        "Reliability": "WIFI_RLBLTY_SCR",
        "Speed": "WIFI_SPEED_SCR",
        "Connection effort": "WIFI_CNCT_EFFRT_SCR",
        "Price": "WIFI_PRICE_SCR",
    }

    metric_scale_cutoff = 75.25 if (df[list(metric_map.values())].stack().quantile(0.99) > 5.5) else 4
    metric_low_cutoff = 25.75 if metric_scale_cutoff == 75.25 else 2

    for metric_name, metric_col in metric_map.items():
        if metric_col not in df.columns:
            continue

        low_avg_nps = df.loc[df[metric_col] <= metric_low_cutoff, "LIKELIHOOD_RECOMMEND_SCL"].mean()
        high_avg_nps = df.loc[df[metric_col] >= metric_scale_cutoff, "LIKELIHOOD_RECOMMEND_SCL"].mean()
        gap_rows.append(
            {
                "Metric": metric_name,
                "Avg_NPS_When_Low_Satisfaction": low_avg_nps,
                "Avg_NPS_When_High_Satisfaction": high_avg_nps,
                "Gap": high_avg_nps - low_avg_nps,
            }
        )

    if gap_rows:
        gap_df = pd.DataFrame(gap_rows).round(2)
        st.table(style_table(gap_df, precision=2))
        render_caption(
            "This table compares customers reporting very poor experiences against customers reporting very positive experiences for each WiFi attribute. The Gap column measures how much overall customer recommendation behavior changes between those two groups.\n"
            "A larger Gap could suggest a stronger relationship between that WiFi attribute and overall customer satisfaction. However, there could be other factors determining a customer's NPS score" 
        )
    else:
        st.info("No metric columns available for low-vs-high NPS gap table.")
else:
    st.info("LIKELIHOOD_RECOMMEND_SCL is required for low-vs-high NPS gap analysis.")


st.markdown("---")
# st.caption("Tip: use sidebar filters to isolate cabins, fleets, and NPS categories for focused analysis.")


st.markdown("### Qualitative Wi-Fi Analysis")

if not enable_sentiment:
    st.info("Enable qualitative data in the sidebar to include nps_sentiment.csv insights.")
elif df_sentiment is None or df_sentiment.empty:
    st.info("No qualitative records available for current filters.")
else:
    qual_total = len(df_sentiment)
    coverage_pct = (qual_total / len(df) * 100) if len(df) > 0 else np.nan

    q0, q1, q2 = st.columns(3)
    with q0:
        metric_card("Total Comments", format_number(total_comment_rows))
    with q1:
        metric_card("WiFi Comments", format_number(qual_total))
    with q2:
        metric_card("% WiFi Comments", f"{coverage_pct:.2f}%" if not pd.isna(coverage_pct) else "NA")


    category_col = WIFI_CATEGORY_COL if WIFI_CATEGORY_COL in df_sentiment.columns else None
    if category_col is None:
        fallback = [c for c in df_sentiment.columns if "category" in c.lower() and "wifi" in c.lower()]
        category_col = fallback[0] if fallback else None

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if category_col is not None:
            category_counts = (
                df_sentiment[category_col]
                .fillna("UNKNOWN")
                .value_counts()
                .reset_index()
            )
            category_counts.columns = ["WiFi_Category", "Count"]

            # Use compact category codes on x-axis and keep full names in legend/mapping.
            category_counts = category_counts.sort_values("Count", ascending=False).reset_index(drop=True)
            category_counts["Category_Code"] = [f"C{i + 1}" for i in range(len(category_counts))]

            fig_cat = px.bar(
                category_counts,
                x="Category_Code",
                y="Count",
                color="WiFi_Category",
                title="Wi-Fi Comment Category Distribution",
                hover_data={"WiFi_Category": True, "Category_Code": False, "Count": True},
            )
            fig_cat.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=55, b=10),
                xaxis_title="Category Code",
                yaxis_title="Comment Count",
                legend_title_text="Wi-Fi Category",
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            render_caption(
                "This chart groups comments into common WiFi-related themes. Categories with larger volumes indicate issues or experiences mentioned most frequently by customers and may represent the most visible areas for improvement."
            )

            # with st.expander("Category Code Mapping"):
            #     st.dataframe(
            #         category_counts[["Category_Code", "WiFi_Category", "Count"]],
            #         use_container_width=True,
            #         hide_index=True,
            #     )
        else:
            st.info("Wi-Fi category column not found in sentiment data.")

    with c2:
        if "NET_PROMO_CATG" in df_sentiment.columns:
            promo_mix = (
                df_sentiment["NET_PROMO_CATG"]
                .fillna("UNKNOWN")
                .astype(str)
                .str.upper()
                .value_counts()
                .reset_index()
            )
            promo_mix.columns = ["NET_PROMO_CATG", "Count"]
            fig_promo_mix = px.pie(
                promo_mix,
                names="NET_PROMO_CATG",
                values="Count",
                title="NPS Mix in WiFi Comments",
                hole=0.45,
                color_discrete_sequence=["#0B5FFF", "#00A3A3", "#0A2540", "#A8DADC"],
            )
            fig_promo_mix.update_layout(height=400, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig_promo_mix, use_container_width=True)
            render_caption(
                "This chart groups comments into common WiFi-related themes. Categories with larger volumes indicate issues or experiences mentioned most frequently by customers and may represent the most visible areas for improvement."
            )

    if category_col is not None and "NET_PROMO_CATG" in df_sentiment.columns:
        category_nps_table = df_sentiment.copy()
        category_nps_table[category_col] = category_nps_table[category_col].fillna("UNKNOWN").astype(str)
        category_nps_table["NET_PROMO_CATG"] = (
            category_nps_table["NET_PROMO_CATG"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
        )

        category_nps_summary = (
            category_nps_table.groupby(category_col, dropna=False)
            .agg(
                Comment_Count=(category_col, "size"),
                Promoter_Pct=("NET_PROMO_CATG", lambda s: (s == "PROMOTER").mean() * 100),
                Passive_Pct=("NET_PROMO_CATG", lambda s: (s == "PASSIVE").mean() * 100),
                Detractor_Pct=("NET_PROMO_CATG", lambda s: (s == "DETRACTOR").mean() * 100),
            )
            .reset_index()
            .rename(columns={category_col: "WiFi_Category"})
            .sort_values("Comment_Count", ascending=False)
        )

        st.markdown("#### Category x NPS Mix")
        st.dataframe(category_nps_summary.round(2), use_container_width=True, hide_index=True)
        render_caption(
            "This table combines customer sentiment with WiFi complaint themes. It helps identify which WiFi issues are most commonly associated with promoters, passives, or detractors and can be used to prioritize improvement efforts."
        )

    if TEXT_COL in df_sentiment.columns:
        st.markdown("#### Comment Explorer")

        show_cols = [c for c in ["SEG_DEP_DT", "NET_PROMO_CATG", "FLEET_CD", "SUBFLEET_CD", "CABIN_FLOWN_CD", TEXT_COL] if c in df_sentiment.columns]

        if category_col is not None:
            cat_options = ["All"] + sorted(df_sentiment[category_col].dropna().astype(str).unique().tolist())
            picked_cat = st.selectbox("Filter comments by Wi-Fi category", options=cat_options)
        else:
            picked_cat = "All"

        comments_view = df_sentiment.copy()
        if category_col is not None and picked_cat != "All":
            comments_view = comments_view[comments_view[category_col].astype(str) == picked_cat]

        st.dataframe(comments_view[show_cols].head(200), use_container_width=True, hide_index=True)