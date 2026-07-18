from __future__ import annotations

import plotly.express as px
import streamlit as st

from ai_assistant import MerchantMateAssistant
from analysis import (
    build_kpis,
    cancellation_summary,
    monthly_revenue_trend,
    revenue_by_country,
    segment_summary,
    top_products_by_revenue,
)
from data_loader import CURRENCY_SYMBOL, clean_retail_data

st.set_page_config(page_title="MerchantMate", page_icon="shopping_trolley", layout="wide")

FIXED_SEGMENT_COUNT = 4


@st.cache_data
def load_dataset() -> object:
    return clean_retail_data()


@st.cache_data
def compute_dashboard_data() -> dict:
    df = load_dataset()
    return {
        "df": df,
        "kpis": build_kpis(df),
        "products": top_products_by_revenue(df),
        "countries": revenue_by_country(df),
        "monthly": monthly_revenue_trend(df),
        "segments": segment_summary(df, n_clusters=FIXED_SEGMENT_COUNT),
        "cancellations": cancellation_summary(df),
    }


def currency(value: float) -> str:
    return f"{CURRENCY_SYMBOL}{value:,.2f}"


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "I am **MerchantMate**, your retail copilot. Ask me about sales, products, countries, "
                    "customer segments, cancellations, or trends in the dataset."
                ),
            }
        ]
    if "show_dashboard" not in st.session_state:
        st.session_state.show_dashboard = False
    if "dashboard_requested" not in st.session_state:
        st.session_state.dashboard_requested = False
    if "follow_up_prompts" not in st.session_state:
        st.session_state.follow_up_prompts = []


def clear_chat() -> None:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "I am **MerchantMate**, your retail copilot. Ask me about sales, products, countries, "
                "customer segments, cancellations, or trends in the dataset."
            ),
        }
    ]
    st.session_state.show_dashboard = False
    st.session_state.dashboard_requested = False
    st.session_state.follow_up_prompts = []


def toggle_dashboard() -> None:
    st.session_state.show_dashboard = not st.session_state.show_dashboard


def submit_question(question: str, assistant: MerchantMateAssistant, df) -> None:
    cleaned = question.strip()
    if not cleaned:
        return

    st.session_state.messages.append({"role": "user", "content": cleaned})
    with st.chat_message("assistant"):
        with st.spinner("MerchantMate is reviewing the data..."):
            answer = assistant.answer_question(cleaned, n_clusters=FIXED_SEGMENT_COUNT, df=df)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.follow_up_prompts = assistant.follow_up_prompts(cleaned)
    st.session_state.dashboard_requested = assistant.should_show_dashboard(cleaned)
    if not st.session_state.dashboard_requested:
        st.session_state.show_dashboard = False


ensure_session_state()
assistant = MerchantMateAssistant()
payload = compute_dashboard_data()
df = payload["df"]
kpis = payload["kpis"]
cancellations = payload["cancellations"]

st.title("MerchantMate")
st.caption(
    "Assistant-first retail intelligence. Chat first. Supporting dashboard only appears when you ask for it."
)

st.sidebar.header("Conversation")
st.sidebar.button("Clear chat", on_click=clear_chat, use_container_width=True)
for idx, message in enumerate(st.session_state.messages):
    speaker = "You" if message["role"] == "user" else "MerchantMate"
    preview = message["content"].replace("\n", " ")
    st.sidebar.caption(f"{speaker} {idx + 1}")
    st.sidebar.write(preview[:120] + ("..." if len(preview) > 120 else ""))

hero_left, hero_right = st.columns([1.2, 0.8], vertical_alignment="top")
with hero_left:
    st.markdown("### Retail copilot")
    st.write(
        "Ask MerchantMate in plain language. It works like a business-friendly copilot for **sales**, "
        "**products**, **countries**, **customer segments**, **cancellations**, and **retail trends**."
    )
with hero_right:
    st.info(
        "Dataset limit: MerchantMate can explain **revenue** and **customer behavior**, but not true "
        "**profit margins**, because the file has no product cost data."
    )

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total Revenue", currency(kpis["total_revenue_gbp"]))
metric_col2.metric("Known Customers", f"{kpis['unique_customers']:,}")
metric_col3.metric("Cancellation Rate", f"{kpis['cancellation_rate_pct']:.2f}%")

st.markdown("### Conversation")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.follow_up_prompts:
    st.markdown("### Suggested next asks")
    follow_cols = st.columns(len(st.session_state.follow_up_prompts))
    for idx, prompt in enumerate(st.session_state.follow_up_prompts):
        if follow_cols[idx].button(prompt, key=f"follow_up_{idx}", use_container_width=True):
            submit_question(prompt, assistant, df)
            st.rerun()

user_input = st.chat_input("Ask MerchantMate about the retail dataset")
if user_input:
    submit_question(user_input, assistant, df)
    st.rerun()

if not assistant.is_configured():
    st.warning(
        "The Groq API key is missing or invalid. MerchantMate will fall back to simple local responses "
        "until the key is available."
    )

if st.session_state.dashboard_requested:
    dashboard_label = "Hide supporting dashboard" if st.session_state.show_dashboard else "Open supporting dashboard"
    st.button(dashboard_label, key="dashboard_toggle_text", on_click=toggle_dashboard, type="tertiary")

if st.session_state.show_dashboard:
    st.markdown("### Supporting dashboard")
    st.caption("This dashboard is supporting evidence for the chart, graph, table, trend, or dashboard request in your latest question.")

    with st.expander("Dashboard controls", expanded=False):
        show_dataset = st.toggle("Show dataset preview", value=False, key="show_dataset_toggle")
        st.write("Customer segments are fixed at **4** in this prototype.")

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Total Revenue", currency(kpis["total_revenue_gbp"]))
    metric_col2.metric("Completed Orders", f"{kpis['completed_orders']:,}")
    metric_col3.metric("Known Customers", f"{kpis['unique_customers']:,}")
    metric_col4.metric("Avg Order Value", currency(kpis["average_order_value_gbp"]))
    metric_col5.metric("Cancellation Rate", f"{kpis['cancellation_rate_pct']:.2f}%")

    st.markdown("### Revenue trend")
    trend_fig = px.line(
        payload["monthly"],
        x="OrderMonth",
        y="RevenueGBP",
        markers=True,
        title="Monthly revenue trend",
    )
    trend_fig.update_layout(yaxis_title="Revenue (GBP)", xaxis_title="Month")
    st.plotly_chart(trend_fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Top products")
        product_fig = px.bar(
            payload["products"].sort_values("RevenueGBP"),
            x="RevenueGBP",
            y="Description",
            orientation="h",
            title="Top products by revenue",
        )
        product_fig.update_layout(yaxis_title="Product", xaxis_title="Revenue (GBP)")
        st.plotly_chart(product_fig, use_container_width=True)

    with right:
        st.markdown("### Top countries")
        country_fig = px.bar(
            payload["countries"],
            x="Country",
            y="RevenueGBP",
            title="Revenue by country",
        )
        country_fig.update_layout(yaxis_title="Revenue (GBP)", xaxis_title="Country")
        st.plotly_chart(country_fig, use_container_width=True)

    segment_col, cancel_col = st.columns([1.3, 0.7], vertical_alignment="top")
    with segment_col:
        st.markdown("### Customer segments")
        st.dataframe(payload["segments"], use_container_width=True)
    with cancel_col:
        st.markdown("### Cancellation snapshot")
        st.metric("Cancellation invoices", f"{cancellations['cancellation_invoices']:,}")
        st.metric("Cancellation rows", f"{cancellations['cancellation_rows']:,}")
        st.metric("Cancellation value", currency(abs(cancellations["cancellation_revenue_gbp"])))

    if st.session_state.get("show_dataset_toggle", False):
        st.markdown("### Dataset preview")
        st.dataframe(
            df[
                [
                    "InvoiceNo",
                    "Description",
                    "Quantity",
                    "UnitPrice",
                    "RevenueGBP",
                    "Country",
                    "IsCancellation",
                ]
            ].head(100),
            use_container_width=True,
        )
