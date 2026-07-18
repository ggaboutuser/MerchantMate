from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "Online Retail.xlsx"
CURRENCY_CODE = "GBP"
CURRENCY_SYMBOL = "GBP "


def load_raw_data(dataset_path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the original Excel dataset without applying business rules yet."""
    return pd.read_excel(Path(dataset_path))


def clean_retail_data(dataset_path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Normalize types and add retail-specific helper columns for analysis."""
    df = load_raw_data(dataset_path).copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].fillna("Unknown Product").astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df["IsCancellation"] = df["InvoiceNo"].str.startswith("C")
    df["IsReturn"] = df["Quantity"] < 0
    df["HasCustomerID"] = df["CustomerID"].notna()
    df["RevenueGBP"] = df["Quantity"] * df["UnitPrice"]
    df["AbsRevenueGBP"] = df["RevenueGBP"].abs()
    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["OrderDate"] = df["InvoiceDate"].dt.date
    df["DataQualityFlag"] = "ok"
    df.loc[df["UnitPrice"] <= 0, "DataQualityFlag"] = "non_positive_price"
    df.loc[df["InvoiceDate"].isna(), "DataQualityFlag"] = "missing_invoice_date"

    return df


def get_completed_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that represent completed, positive-value sales."""
    return df.loc[
        (~df["IsCancellation"])
        & (~df["IsReturn"])
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ].copy()


def get_customer_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Completed sales that can be tied back to a known customer."""
    completed = get_completed_sales(df)
    return completed.loc[completed["HasCustomerID"]].copy()


def build_dataset_profile(df: pd.DataFrame) -> dict:
    completed = get_completed_sales(df)
    return {
        "row_count": int(len(df)),
        "completed_sales_rows": int(len(completed)),
        "missing_customer_rows": int(df["CustomerID"].isna().sum()),
        "cancellation_rows": int(df["IsCancellation"].sum()),
        "return_rows": int(df["IsReturn"].sum()),
        "countries": int(df["Country"].nunique()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
        "currency": CURRENCY_CODE,
    }
