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

import energy_gap.data_checks as data_checks
import energy_gap.load as load


# ---------------------------------------------------------------------------
# Set this to True for a fast test on the Fox site only (137 buildings, ~15s).
# Set it back to False to load everything (1,578 buildings, ~2 minutes).
TEST_ONLY = False
# ---------------------------------------------------------------------------


def main():
    start = time.time()
    mode = "TEST (Fox site only)" if TEST_ONLY else "FULL (all sites)"

    print("=" * 60)
    print(f"  Loading the BDG2 dataset into Postgres  --  {mode}")
    print("=" * 60)

    # Step 1. Create the tables. This wipes any existing ones, which is why
    # re-running this file is always safe: it starts from a clean database
    # rather than trying to add rows on top of rows already there.
    print("\n[1/5] Creating tables...")
    load.init_db()

    print("\n[2/5] Loading building records...")
    load.load_metadata()

    print("\n[3/5] Loading weather...")
    load.load_weather()

    print("\n[4/5] Loading meter readings (the slow one)...")
    load.load_meters(sites=["Fox"] if TEST_ONLY else None)

    print("\n[5/5] Checking the loaded data...")
    data_checks.check_data()

    mins, secs = divmod(int(time.time() - start), 60)
    print("=" * 60)
    print(f"  Done in {mins}m {secs}s")
    if TEST_ONLY:
        print("  This was the test run. Set TEST_ONLY = False above and")
        print("  press Run again to load all 1,578 buildings.")
    else:
        print("  Expected: 1,636 buildings / 331,166 weather / 25,212,579 readings")
        print("  All 19 sites should read 'ok' in check 3.")
    print("=" * 60)


# Only run when this file is executed directly, not when something imports it.
if __name__ == "__main__":
    main()