// The only file that talks to Postgres. Everything else asks this one.
// The Node counterpart of pipeline/energy_gap/db.py.

import { fileURLToPath } from "node:url";
import path from "node:path";
import dotenv from "dotenv";
import pg from "pg";

// Read the same .env at the repo root that the Python pipeline reads. One
// connection string, one place to change it.
const here = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(here, "../../.env") });

if (!process.env.DATABASE_URL) {
  throw new Error(
    "DATABASE_URL is not set. Check that .env exists at the repo root."
  );
}

// A pool, not a single connection. Every browser request needs a connection,
// and opening a fresh one each time is slow. The pool keeps a few open and
// hands them out, which is why an API can answer many requests at once.
export const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  max: 10,
  idleTimeoutMillis: 30000,
});

// Send one SQL statement to Postgres and return just the rows.
//await` pauses this function until the answer comes back, letting the server
// handle other requests meanwhile
export async function query(text, params) {
  const result = await pool.query(text, params);
  return result.rows;
}

// The run_id of the most recent pipeline run
export async function latestRunId() {
  const rows = await query("SELECT max(run_id) AS run_id FROM model_runs");
  return rows[0]?.run_id ?? null;
}