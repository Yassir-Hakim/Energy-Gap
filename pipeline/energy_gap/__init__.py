"""Energy performance gap pipeline.

This file marks the folder as an importable Python package. Without it,
`from energy_gap.load import ...` fails, so run.py and the tests would both
break. It is intentionally almost empty.

The modules, in the order they depend on each other:
    config.py   every threshold, path and setting. Imports nothing.
    db.py       the only module that talks to Postgres. Needs config.
    load.py     reads the CSVs and reshapes them. Needs config and db.
    checks.py   four read-only sanity checks. Needs config and db.
    metrics.py  the ASHRAE Guideline 14 statistics. Standalone.
"""