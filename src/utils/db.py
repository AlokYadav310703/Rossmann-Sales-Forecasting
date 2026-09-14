"""
Shared database connection helper.
Import this from any notebook or script instead of repeating
create_engine(...) / pd.read_sql(...) boilerplate everywhere.
"""

import pandas as pd
from sqlalchemy import create_engine

from src.config import DATABASE_URL


def get_engine(database_url: str = None):
    """Create a SQLAlchemy engine. Defaults to DATABASE_URL from config/.env."""
    return create_engine(database_url or DATABASE_URL)


def run_query(query: str, engine=None) -> pd.DataFrame:
    """Run a SQL query string and return the result as a DataFrame."""
    engine = engine or get_engine()
    return pd.read_sql(query, engine)


def run_query_from_file(filepath: str, engine=None) -> pd.DataFrame:
    """Read a .sql file and execute its contents, returning a DataFrame.
    Assumes the file contains a single SELECT query."""
    engine = engine or get_engine()
    with open(filepath, "r") as f:
        query = f.read()
    return pd.read_sql(query, engine)
