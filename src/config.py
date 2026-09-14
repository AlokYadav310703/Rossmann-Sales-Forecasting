"""
Central configuration for the project.
Reads sensitive values (DB credentials, etc.) from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (never commit .env to git)
load_dotenv()

# --- Database ---
# Example (Supabase Postgres, session pooler - used for batch loads/scripts):
# postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/processed/rossmann.db")

# --- Paths ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"

STORE_CSV_PATH = os.path.join(RAW_DATA_DIR, "store.csv")
TRAIN_CSV_PATH = os.path.join(RAW_DATA_DIR, "train.csv")

# --- Modeling ---
RANDOM_SEED = 42
DEFAULT_FORECAST_HORIZON_DAYS = 14
