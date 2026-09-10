"""Press Run on this file to load the whole dataset into Postgres.

To do a quick 15-second test on one site first, leave TEST_ONLY as True.
Set it to False for the real run.

This file is the only way the pipeline is started. It calls the four working
modules in order:
    load.init_db()        creates the tables      (sql/001_schema.sql)
    load.load_metadata()  1,636 building records  (load.py)
    load.load_weather()   331,166 weather rows    (load.py)
    load.load_meters()    25.2m meter readings    (load.py)
    checks.check_data()   four sanity checks      (checks.py)
"""

import sys
import time
from pathlib import Path

# This file sits in pipeline/, next to the energy_gap folder. Adding pipeline/
# to Python's search path means the Run button works from anywhere, which is
# the thing that made running a file directly fail before.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from energy_gap import data_checks, load, score   # noqa: E402


# ---------------------------------------------------------------------------
# Set this to True for a fast test on the Fox site only (137 buildings, ~15s).
# Set it back to False to load everything (1,578 buildings, ~2 minutes).
TEST_ONLY = False

# Load the data, or fit the models, or both. Loading only needs doing once;
# after that set LOAD_DATA = False and re-run scoring as often as you like.
LOAD_DATA = True
SCORE_MODELS = True

# Cap the number of buildings scored. Set to 20 for a trial run (about 30
# seconds); set to None to score everything. Scoring is the slow part, roughly
# 1.5 seconds per building, so the full set takes 30-45 minutes.
BUILDING_LIMIT = None
N_JOBS = -1
# ---------------------------------------------------------------------------


def main():
    start = time.time()
    mode = "TEST (Fox site only)" if TEST_ONLY else "FULL (all sites)"
    sites = ["Fox"] if TEST_ONLY else None

    print("=" * 60)
    print(f"  Energy performance gap pipeline  --  {mode}")
    print("=" * 60)

    if LOAD_DATA:
        # Step 1 wipes the tables first, which is why re-running this file is
        # always safe: it starts from a clean database rather than adding rows
        # on top of rows already there.
        print("\n[1/5] Creating tables...")
        load.init_db()

        print("\n[2/5] Loading building records...")
        load.load_metadata()

        print("\n[3/5] Loading weather...")
        load.load_weather()

        print("\n[4/5] Loading meter readings (the slow one)...")
        load.load_meters(sites=sites)

        print("\n[5/5] Checking the loaded data...")
        data_checks.check_data()

    if SCORE_MODELS:
        n = BUILDING_LIMIT if BUILDING_LIMIT else "all"
        print(f"\n[models] Fitting both models on {n} buildings...")
        run_id = score.run(limit=BUILDING_LIMIT, sites=sites, n_jobs=N_JOBS,
                   notes=f"{mode}, limit={BUILDING_LIMIT}")
        score.summarise(run_id)

    mins, secs = divmod(int(time.time() - start), 60)
    print("=" * 60)
    print(f"  Done in {mins}m {secs}s")
    if TEST_ONLY:
        print("  Test run. Set TEST_ONLY = False for all 1,578 buildings.")
    if BUILDING_LIMIT:
        print(f"  Only {BUILDING_LIMIT} buildings scored. Set BUILDING_LIMIT = None")
        print("  to score everything (30-45 minutes).")
    print("=" * 60)


# Only run when this file is executed directly, not when something imports it.
if __name__ == "__main__":
    main()

