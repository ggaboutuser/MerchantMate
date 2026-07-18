from __future__ import annotations

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_loader import CURRENCY_CODE, build_dataset_profile, get_completed_sales, get_customer_sales


SEGMENT_NAME_POOL = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "At Risk",
    "Occasional Buyers",
    "Needs Attention",
]


def build_kpis(df: pd.DataFrame) -> dict[str, Any]:
    completed = get_completed_sales(df)
    customer_sales = get_customer_sales(df)

    total_revenue = float(completed["RevenueGBP"].sum())
    unique_invoices = int(completed["InvoiceNo"].nunique())
    unique_customers = int(customer_sales["CustomerID"].nunique())
    average_order_value = total_revenue / unique_invoices if unique_invoices else 0.0
    cancellation_rate = float(df["IsCancellation"].mean() * 100)

    return {
        "currency": CURRENCY_CODE,
        "total_revenue_gbp": total_revenue,
        "completed_orders": unique_invoices,
        "unique_customers": unique_customers,
        "average_order_value_gbp": average_order_value,
        "cancellation_rate_pct": cancellation_rate,
        "profile": build_dataset_profile(df),
    }


def top_products_by_revenue(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    completed = get_completed_sales(df)
    summary = (
        completed.groupby("Description", as_index=False)
        .agg(
            RevenueGBP=("RevenueGBP", "sum"),
            QuantitySold=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("RevenueGBP", ascending=False)
        .head(limit)
    )
    return summary


def revenue_by_country(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    completed = get_completed_sales(df)
    summary = (
        completed.groupby("Country", as_index=False)
        .agg(
            RevenueGBP=("RevenueGBP", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
        )
        .sort_values("RevenueGBP", ascending=False)
        .head(limit)
    )
    return summary


def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    completed = get_completed_sales(df)
    summary = (
        completed.groupby("OrderMonth", as_index=False)
        .agg(
            RevenueGBP=("RevenueGBP", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("OrderMonth")
    )
    return summary


def cancellation_summary(df: pd.DataFrame) -> dict[str, Any]:
    cancellation_rows = df.loc[df["IsCancellation"]].copy()
    impacted_revenue = float(cancellation_rows["RevenueGBP"].sum())
    return {
        "cancellation_rows": int(len(cancellation_rows)),
        "cancellation_invoices": int(cancellation_rows["InvoiceNo"].nunique()),
        "cancellation_revenue_gbp": impacted_revenue,
    }


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    customer_sales = get_customer_sales(df)
    if customer_sales.empty:
        return pd.DataFrame(columns=["CustomerID", "Recency", "Frequency", "Monetary", "AvgOrderValue"])

    snapshot_date = customer_sales["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        customer_sales.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda dates: (snapshot_date - dates.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("RevenueGBP", "sum"),
        )
        .reset_index()
    )
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    return rfm.sort_values("Monetary", ascending=False)


def build_customer_segments(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    rfm = build_rfm_table(df)
    if rfm.empty:
        return rfm

    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    rfm["Cluster"] = model.fit_predict(scaled)

    centroids = (
        rfm.groupby("Cluster", as_index=False)
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
        )
        .sort_values(["Monetary", "Frequency", "Recency"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    ordered_clusters = centroids["Cluster"].tolist()
    labels = SEGMENT_NAME_POOL[: len(ordered_clusters)]
    cluster_to_label = {cluster: labels[idx] for idx, cluster in enumerate(ordered_clusters)}
    rfm["Segment"] = rfm["Cluster"].map(cluster_to_label)

    return rfm.sort_values(["Cluster", "Monetary"], ascending=[True, False])


def segment_summary(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    segments = build_customer_segments(df, n_clusters=n_clusters)
    if segments.empty:
        return segments

    summary = (
        segments.groupby("Segment", as_index=False)
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .sort_values("AvgMonetary", ascending=False)
    )
    return summary


def top_segment(df: pd.DataFrame, n_clusters: int = 4) -> dict[str, Any]:
    summary = segment_summary(df, n_clusters=n_clusters)
    if summary.empty:
        return {}
    return summary.iloc[0].to_dict()


def build_question_context(df: pd.DataFrame, n_clusters: int = 4) -> dict[str, Any]:
    kpis = build_kpis(df)
    top_products = top_products_by_revenue(df, limit=5).to_dict(orient="records")
    countries = revenue_by_country(df, limit=5).to_dict(orient="records")
    monthly = monthly_revenue_trend(df).tail(6).to_dict(orient="records")
    segments = segment_summary(df, n_clusters=n_clusters).to_dict(orient="records")
    cancellations = cancellation_summary(df)

    return {
        "kpis": kpis,
        "top_products": top_products,
        "top_countries": countries,
        "recent_monthly_trend": monthly,
        "segment_summary": segments,
        "cancellations": cancellations,
        "notes": [
            "All revenue values are in GBP.",
            "Revenue is computed as Quantity multiplied by UnitPrice.",
            "Completed sales exclude cancellation invoices, returns, and non-positive prices.",
            "The dataset does not include cost-of-goods, so the project can explain sales revenue but not true profit.",
        ],
    }


def recent_sales_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    completed = get_completed_sales(df)
    if completed.empty:
        return {"latest_month_revenue_gbp": 0.0, "latest_month_orders": 0}

    latest_month = completed["OrderMonth"].max()
    latest_rows = completed.loc[completed["OrderMonth"] == latest_month]
    return {
        "latest_month": latest_month,
        "latest_month_revenue_gbp": float(latest_rows["RevenueGBP"].sum()),
        "latest_month_orders": int(latest_rows["InvoiceNo"].nunique()),
    }


def data_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "missing_customer_rows": int(df["CustomerID"].isna().sum()),
        "missing_description_rows": int((df["Description"] == "Unknown Product").sum()),
        "non_positive_price_rows": int((df["UnitPrice"] <= 0).sum()),
        "return_rows": int(df["IsReturn"].sum()),
        "cancellation_rows": int(df["IsCancellation"].sum()),
    }


def build_targeted_context(topic: str, df: pd.DataFrame, n_clusters: int = 4) -> dict[str, Any]:
    base = {
        "kpis": build_kpis(df),
        "profile": build_dataset_profile(df),
        "notes": [
            "All revenue values are in GBP.",
            "Revenue is computed as Quantity multiplied by UnitPrice.",
            "Completed sales exclude cancellation invoices, returns, and non-positive prices.",
            "The dataset does not include cost-of-goods, so the project can explain sales revenue but not true profit.",
        ],
    }

    if topic == "products":
        base["top_products"] = top_products_by_revenue(df, limit=10).to_dict(orient="records")
    elif topic == "countries":
        base["top_countries"] = revenue_by_country(df, limit=10).to_dict(orient="records")
    elif topic == "segments":
        base["segment_summary"] = segment_summary(df, n_clusters=n_clusters).to_dict(orient="records")
        base["segment_sample"] = build_customer_segments(df, n_clusters=n_clusters).head(20).to_dict(orient="records")
    elif topic == "cancellations":
        base["cancellations"] = cancellation_summary(df)
        base["data_quality"] = data_quality_summary(df)
    elif topic == "trends":
        base["monthly_trend"] = monthly_revenue_trend(df).to_dict(orient="records")
        base["recent_snapshot"] = recent_sales_snapshot(df)
    elif topic == "data_quality":
        base["data_quality"] = data_quality_summary(df)
    else:
        return build_question_context(df, n_clusters=n_clusters)

    return base
