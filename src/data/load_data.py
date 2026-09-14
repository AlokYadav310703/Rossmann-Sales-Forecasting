"""
Loads store.csv and train.csv from data/raw/, cleans column names/types
to match the schema defined in sql/schema.sql, and writes them into the
database specified by DATABASE_URL (Supabase Postgres, or local SQLite fallback).

Run with:
    python -m src.data.load_data
"""

import pandas as pd
from sqlalchemy import create_engine

from src.config import DATABASE_URL, STORE_CSV_PATH, TRAIN_CSV_PATH


def load_stores(engine) -> None:
    """Load store.csv into the `stores` table."""
    df = pd.read_csv(STORE_CSV_PATH)

    # Rename CSV columns -> DB schema column names
    df = df.rename(columns={
        "Store": "store_id",
        "StoreType": "store_type",
        "Assortment": "assortment",
        "CompetitionDistance": "competition_distance",
        "CompetitionOpenSinceMonth": "competition_open_since_month",
        "CompetitionOpenSinceYear": "competition_open_since_year",
        "Promo2": "promo2",
        "Promo2SinceWeek": "promo2_since_week",
        "Promo2SinceYear": "promo2_since_year",
        "PromoInterval": "promo_interval",
    })

    # Type cleanup
    df["promo2"] = df["promo2"].astype(bool).astype(object)

    df.to_sql("stores", engine, if_exists="append", index=False)
    print(f"Loaded {len(df)} rows into 'stores'")


def load_sales(engine, chunksize: int = 50_000) -> None:
    """Load train.csv into the `sales_daily` table, in chunks (it's ~1M rows)."""
    total_rows = 0

    for chunk in pd.read_csv(TRAIN_CSV_PATH, parse_dates=["Date"], chunksize=chunksize):
        chunk = chunk.rename(columns={
            "Store": "store_id",
            "DayOfWeek": "day_of_week",
            "Date": "date",
            "Sales": "sales",
            "Customers": "customers",
            "Open": "open",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday",
        })

        # Type cleanup
        chunk["open"] = chunk["open"].astype(bool).astype(object)
        chunk["promo"] = chunk["promo"].astype(bool).astype(object)
        chunk["school_holiday"] = chunk["school_holiday"].astype(bool).astype(object)
        # StateHoliday mixes 0 (int) and 'a'/'b'/'c' (str) in the raw CSV -> force to string
        chunk["state_holiday"] = chunk["state_holiday"].astype(str)

        chunk.to_sql("sales_daily", engine, if_exists="append", index=False)
        total_rows += len(chunk)
        print(f"  ...loaded {total_rows} rows so far")

    print(f"Finished loading {total_rows} rows into 'sales_daily'")


def main():
    engine = create_engine(DATABASE_URL)

    print(f"Connecting to: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")

    print("\nLoading stores...")
    load_stores(engine)

    print("\nLoading sales_daily (this may take a few minutes)...")
    load_sales(engine)

    print("\nDone.")


if __name__ == "__main__":
    main()
