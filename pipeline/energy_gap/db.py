"""Opens the database connection and provides the fast bulk-load helper.

The only module that imports psycopg. Everything else asks this one, so if the
connection details ever change there is exactly one file to edit.
"""

import io
import os

import pandas as pd
import psycopg
from dotenv import load_dotenv

from energy_gap.config import REPO_ROOT

# Read <repo>/.env so DATABASE_URL becomes available as an environment variable.
load_dotenv(REPO_ROOT / ".env")


def get_conn():
    """Open a connection to Postgres using the URL from .env.

    Used as `with get_conn() as conn:` so the connection always closes itself,
    even if something fails partway through.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Check that .env exists at the repo root "
            "and contains a line like:\n"
            "  DATABASE_URL=postgresql://energy:energy@localhost:5432/energygap"
        )
    return psycopg.connect(url)


def run_sql_file(path):
    """Run every statement in a .sql file. Used once, to create the tables."""
    sql = open(path).read()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()   # nothing is permanent until you commit


def copy_frame(df, table, columns):
    """Bulk-load a DataFrame into a table using Postgres COPY.

    pandas writes the frame to CSV text in memory (never to disk), then COPY
    streams that text into the table in one operation. INSERT would send one
    statement per row; across 25 million rows that is the difference between
    about two minutes and several hours.

    `columns` must name the table columns in the same order as the frame's.
    """
    if df.empty:
        return 0

    buf = io.StringIO()
    # na_rep="" because in CSV mode Postgres reads an unquoted empty field as NULL.
    df.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)

    collist = ", ".join(columns)
    with get_conn() as conn:
        with conn.cursor() as cur:
            with cur.copy(f"COPY {table} ({collist}) FROM STDIN WITH (FORMAT csv)") as copy:
                copy.write(buf.read())
        conn.commit()
    return len(df)


def execute(sql, params=None):
    """Run a single statement and return any rows it produces."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
        conn.commit()
    return rows


def query_df(sql):
    """Run a query and return the result as a DataFrame.

    Built from a plain cursor rather than pandas.read_sql, which prints a
    warning on every call that it only officially supports SQLAlchemy
    connections.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)