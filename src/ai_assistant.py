from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from analysis import build_kpis, build_targeted_context, cancellation_summary, revenue_by_country, segment_summary, top_products_by_revenue, top_segment
from data_loader import DEFAULT_DATASET_PATH, clean_retail_data
from prompts import SYSTEM_PROMPT, build_user_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

TOPIC_KEYWORDS = {
    "products": ("product", "item", "sku", "stock", "best-selling", "top-selling"),
    "countries": ("country", "countries", "region", "market", "location", "geography"),
    "segments": ("segment", "customer", "buyer", "loyal", "rfm", "champion"),
    "cancellations": ("cancel", "return", "refund"),
    "trends": ("trend", "month", "time", "growth", "decline", "recent", "sales"),
    "data_quality": ("missing", "quality", "clean", "null", "dataset"),
}

FALLBACK_RESPONSES = {
    "products": "I can break down top products, sales by item, and which products drive the most revenue in GBP.",
    "countries": "I can compare countries by revenue, orders, and known customers using the current dataset.",
    "segments": "I can explain the customer segments from RFM clustering and which groups look most valuable or at risk.",
    "cancellations": "I can summarize cancellation and return patterns, but I cannot infer exact refund reasons from this dataset alone.",
    "trends": "I can summarize revenue trends over time and highlight recent months in GBP.",
    "data_quality": "I can describe what is missing or noisy in the dataset and how that affects the analysis.",
    "general": "I can help with sales, products, countries, customer segments, cancellations, and monthly trends from the retail dataset.",
}

FOLLOW_UP_PROMPTS = {
    "products": [
        "Which products should we promote more?",
        "Which products have the highest order count?",
        "Show me the product trend chart.",
    ],
    "countries": [
        "Which country should we focus on next?",
        "Show me the country revenue chart.",
        "Which countries have the most known customers?",
    ],
    "segments": [
        "Which segment is most at risk?",
        "How should we retain the Champions segment?",
        "Show me the customer segment table.",
    ],
    "cancellations": [
        "Show me the cancellation overview.",
        "What data-quality issue stands out most?",
        "How big is the cancellation impact in GBP?",
    ],
    "trends": [
        "Show me the monthly revenue trend.",
        "What happened in the latest month?",
        "Which month looks strongest?",
    ],
    "general": [
        "What are the top-selling products?",
        "Which country drives the most revenue?",
        "Which customer segment should we focus on first?",
    ],
}


class MerchantMateAssistant:
    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv(ENV_PATH)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def is_configured(self) -> bool:
        return self.client is not None

    def classify_question(self, question: str) -> str:
        normalized = question.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in normalized for keyword in keywords):
                return topic
        return "general"

    def follow_up_prompts(self, question: str) -> list[str]:
        topic = self.classify_question(question)
        return FOLLOW_UP_PROMPTS.get(topic, FOLLOW_UP_PROMPTS["general"])

    def should_show_dashboard(self, question: str) -> bool:
        normalized = question.lower()
        dashboard_keywords = (
            "dashboard",
            "chart",
            "graph",
            "plot",
            "visual",
            "table",
            "trend",
            "show me",
        )
        return any(keyword in normalized for keyword in dashboard_keywords)

    def format_answer(self, answer: str) -> str:
        replacements = {
            "GBP": "**GBP**",
            "DOTCOM POSTAGE": "**DOTCOM POSTAGE**",
            "Champions": "**Champions**",
            "Cancellation Rate": "**Cancellation Rate**",
            "\u0141": "**GBP** ",
            "\u00a3": "**GBP** ",
        }
        formatted = answer
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        return formatted

    def _build_fallback_answer(self, topic: str, context: dict) -> str:
        kpis = context.get("kpis", {})
        revenue = kpis.get("total_revenue_gbp", 0.0)
        orders = kpis.get("completed_orders", 0)
        opener = FALLBACK_RESPONSES.get(topic, FALLBACK_RESPONSES["general"])
        return self.format_answer(
            (
            f"{opener} Right now the dataset shows about GBP {revenue:,.2f} in completed revenue "
            f"across {orders:,} completed orders."
            )
        )

    def _direct_answer(self, question: str, df: Any, n_clusters: int) -> str | None:
        normalized = question.lower()

        if "currency" in normalized and ("unitprice" in normalized or "revenue" in normalized or "price" in normalized):
            return self.format_answer(
                (
                "UnitPrice is recorded in GBP, so revenue is also in GBP because revenue is calculated as "
                "Quantity multiplied by UnitPrice."
                )
            )

        if ("top-selling" in normalized or "highest revenue" in normalized or "top product" in normalized) and (
            "product" in normalized or "item" in normalized
        ):
            top_products = top_products_by_revenue(df, limit=1)
            if top_products.empty:
                return "I could not find any completed product sales in the dataset."
            row = top_products.iloc[0]
            return self.format_answer(
                (
                f"The highest-revenue product in the current completed-sales analysis is {row['Description']}, "
                f"with GBP {row['RevenueGBP']:,.2f} in revenue, {int(row['QuantitySold']):,} units sold, "
                f"and {int(row['Orders']):,} orders."
                )
            )

        if "most valuable customer segment" in normalized or (
            "which customer segment" in normalized and "focus" in normalized
        ):
            segment = top_segment(df, n_clusters=n_clusters)
            if not segment:
                return "I could not build customer segments from the current dataset."
            return self.format_answer(
                (
                f"The most valuable segment is {segment['Segment']}. It contains {int(segment['Customers']):,} customers "
                f"with an average monetary value of GBP {segment['AvgMonetary']:,.2f}, average frequency of "
                f"{segment['AvgFrequency']:.2f}, and average recency of {segment['AvgRecency']:.2f} days. "
                "This segment matters because it groups the customers who contribute the most revenue, so it helps "
                "you prioritize retention, loyalty offers, and high-value service."
                )
            )

        if "cancellation rate" in normalized:
            kpis = build_kpis(df)
            cancellations = cancellation_summary(df)
            return self.format_answer(
                (
                f"The cancellation rate is {kpis['cancellation_rate_pct']:.2f}%. "
                f"That comes from {cancellations['cancellation_invoices']:,} cancellation invoices in the dataset."
                )
            )

        if "country" in normalized and ("most revenue" in normalized or "drive the most revenue" in normalized):
            countries = revenue_by_country(df, limit=1)
            if not countries.empty:
                row = countries.iloc[0]
                return self.format_answer(
                    (
                    f"The top country by revenue is {row['Country']}, with GBP {row['RevenueGBP']:,.2f}, "
                    f"{int(row['Orders']):,} completed orders, and {int(row['Customers']):,} known customers."
                    )
                )

        if "known customers" in normalized:
            kpis = build_kpis(df)
            return self.format_answer(f"There are **{kpis['unique_customers']:,}** known customers in the completed-sales analysis.")

        if "total completed revenue" in normalized or "average order value" in normalized:
            kpis = build_kpis(df)
            return self.format_answer(
                (
                f"Total completed revenue is GBP {kpis['total_revenue_gbp']:,.2f}, and average order value is "
                f"GBP {kpis['average_order_value_gbp']:,.2f}."
                )
            )

        if "how many countries" in normalized:
            kpis = build_kpis(df)
            return self.format_answer(f"The dataset covers **{kpis['profile']['countries']:,}** countries.")

        return None

    def answer_question(
        self,
        question: str,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        n_clusters: int = 4,
        df: Any | None = None,
    ) -> str:
        df = df if df is not None else clean_retail_data(dataset_path)
        direct = self._direct_answer(question, df, n_clusters)
        if direct:
            return direct

        topic = self.classify_question(question)
        context = build_targeted_context(topic, df, n_clusters=n_clusters)

        if not self.client:
            return self._build_fallback_answer(topic, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, context)},
                ],
            )
            return self.format_answer(response.choices[0].message.content or "No answer returned.")
        except Exception as exc:
            return self.format_answer(
                (
                f"I could not reach the Groq model just now ({exc}). "
                f"{self._build_fallback_answer(topic, context)}"
                )
            )
