"""Every threshold, path and model setting the pipeline uses. No logic lives here."""

from pathlib import Path

# This file is at <repo>/pipeline/energy_gap/config.py, so go up three levels to
# reach the repo root. Paths then work no matter which folder you run from.
REPO_ROOT = Path(__file__).resolve().parents[2]

BDG2_DIR = REPO_ROOT / "data" / "bdg2" / "data"
METADATA_CSV = BDG2_DIR / "metadata" / "metadata.csv"
WEATHER_CSV = BDG2_DIR / "weather" / "weather.csv"
ELECTRICITY_CSV = BDG2_DIR / "meters" / "cleaned" / "electricity_cleaned.csv"

SCHEMA_SQL = REPO_ROOT / "pipeline" / "sql" / "001_schema.sql"

# --- Periods -----------------------------------------------------------------
BASELINE_YEAR = 2016   # the year models learn from
SCORING_YEAR = 2017    # the year we judge

# --- Model shape -------------------------------------------------------------
# Temperatures (Celsius) where the fitted line is allowed to change slope.
TEMP_KNOTS_C = [5, 10, 15, 20, 25, 30]
WEATHER_FFILL_LIMIT_HOURS = 6   # carry a temperature forward over empty gaps up to this long
RIDGE_ALPHA = 0.01              # tiny penalty, only for numerical stability
GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=20, random_state=0)

# --- Gates -------------------------------------------------------------------
# Measured against the real dataset: 88 of the 1,578 electricity buildings fall
# below MIN_BASELINE_HOURS and will come back as insufficient_data.
MIN_BASELINE_HOURS = 5000
MIN_COVERAGE_PCT = 80.0
G14_HOURLY_CVRMSE_MAX = 30.0     # ASHRAE Guideline 14, hourly criteria
G14_HOURLY_NMBE_ABS_MAX = 10.0
GAP_FLAG_PCT = 10.0              # gap must be at least this big to matter
Z_FLAG = 2.0                     # ...and this significant to be believed
HOLDOUT_EVERY_NTH_WEEK = 4

# Hours in 2016 + 2017 (2016 is a leap year): 731 days x 24.
TOTAL_HOURS = 17544