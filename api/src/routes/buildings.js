// The five endpoints the web app actually reads.
//
// All read-only. The pipeline does every calculation in advance and writes the
// results to four tables; these queries only fetch what is already there. That
// is why a page can load in milliseconds despite 25 million meter readings
// sitting in the same database - nothing here ever touches that table.

import express from "express";
import { latestRunId, query } from "../db.js";

export const router = express.Router();

const MODELS = ["towt", "gbm"];

// Every endpoint takes optional ?run_id= and ?model=. Validating them in one
// place means a bad value gives a clear 400 rather than a confusing empty
// result or a database error.
async function params(req) {
  const model = req.query.model ?? "towt";
  if (!MODELS.includes(model)) {
    const err = new Error(`model must be one of: ${MODELS.join(", ")}`);
    err.status = 400;
    throw err;
  }

  let runId = req.query.run_id ? Number(req.query.run_id) : await latestRunId();
  if (!Number.isInteger(runId)) {
    const err = new Error("run_id must be an integer");
    err.status = 400;
    throw err;
  }
  return { runId, model };
}

// GET /api/summary?run_id=&model=
// Counts per status, for the strip across the top of the buildings screen.
router.get("/summary", async (req, res, next) => {
  try {
    const { runId, model } = await params(req);
    const rows = await query(
      `SELECT status, count(*)::int AS n
       FROM building_scores
       WHERE run_id = $1 AND model = $2
       GROUP BY status`,
      [runId, model]
    );

    const counts = Object.fromEntries(rows.map((r) => [r.status, r.n]));
    res.json({
      run_id: runId,
      model,
      total: rows.reduce((sum, r) => sum + r.n, 0),
      counts,
    });
  } catch (err) {
    next(err);
  }
});

// GET /api/buildings?run_id=&model=
// One row per scored building, joined to its metadata and to the other
// model's verdict.
//
// No paging. About 1,570 rows of small columns is a few hundred kilobytes,
// which the browser can hold comfortably, and having the whole set client-side
// means sorting and filtering are instant with no further requests.
router.get("/buildings", async (req, res, next) => {
  try {
    const { runId, model } = await params(req);
    const other = model === "towt" ? "gbm" : "towt";

    const rows = await query(
      `SELECT s.building_id,
              b.site_id,
              b.primary_use,
              b.sqm,
              -- A building with a steam, hot water or gas meter does not heat
              -- with electricity, so a clean result for it means "no
              -- electricity gap", not "no performance gap".
              NOT (b.has_steam OR b.has_hotwater OR b.has_gas) AS heating_electric,
              s.status,
              s.gap_pct,
              s.gap_kwh,
              s.z_score,
              s.coverage_pct,
              s.baseline_hours,
              m.cvrmse,
              m.nmbe,
              m.cvrmse_holdout,
              m.passes_g14,
              $3::text AS other_model,
              o.status AS other_model_status
       FROM building_scores s
       JOIN buildings b ON b.building_id = s.building_id
       LEFT JOIN building_models m
              ON m.run_id = s.run_id AND m.building_id = s.building_id
             AND m.model = s.model
       LEFT JOIN building_scores o
              ON o.run_id = s.run_id AND o.building_id = s.building_id
             AND o.model = $3
       WHERE s.run_id = $1 AND s.model = $2
       ORDER BY s.gap_pct DESC NULLS LAST`,
      [runId, model, other]
    );
    res.json(rows);
  } catch (err) {
    next(err);
  }
});

// GET /api/buildings/:id?run_id=
// Everything about one building: its metadata, and BOTH models' fit
// statistics and scores. The detail screen shows one model at a time but
// needs the other alongside, so both are sent in a single request.
router.get("/buildings/:id", async (req, res, next) => {
  try {
    const { runId } = await params(req);
    const id = req.params.id;

    const meta = await query(
      `SELECT building_id, site_id, primary_use, sub_primary_use, sqm, sqft,
              year_built, floors, occupants, timezone, heating_type,
              has_electricity, has_chilledwater, has_steam, has_hotwater, has_gas,
              NOT (has_steam OR has_hotwater OR has_gas) AS heating_electric
       FROM buildings WHERE building_id = $1`,
      [id]
    );
    if (meta.length === 0) {
      return res.status(404).json({ error: "building not found" });
    }

    const models = await query(
      `SELECT m.model, m.n_train, m.n_params, m.cvrmse, m.nmbe, m.r2,
              m.cvrmse_holdout, m.nmbe_holdout, m.passes_g14,
              m.resid_sigma, m.resid_rho,
              s.baseline_hours, s.scoring_hours, s.coverage_pct,
              s.actual_kwh, s.predicted_kwh, s.gap_kwh, s.gap_pct,
              s.z_score, s.status
       FROM building_scores s
       LEFT JOIN building_models m
              ON m.run_id = s.run_id AND m.building_id = s.building_id
             AND m.model = s.model
       WHERE s.run_id = $1 AND s.building_id = $2
       ORDER BY s.model`,
      [runId, id]
    );

    res.json({ run_id: runId, building: meta[0], models });
  } catch (err) {
    next(err);
  }
});

// GET /api/buildings/:id/daily?run_id=&model=
// Daily actual against predicted across both years, for the line chart.
// `period` marks which side of the baseline/scoring boundary each day is on.
router.get("/buildings/:id/daily", async (req, res, next) => {
  try {
    const { runId, model } = await params(req);
    const rows = await query(
      `SELECT day, period, actual_kwh, predicted_kwh, n_hours
       FROM daily_summary
       WHERE run_id = $1 AND building_id = $2 AND model = $3
       ORDER BY day`,
      [runId, req.params.id, model]
    );
    res.json(rows);
  } catch (err) {
    next(err);
  }
});

// GET /api/buildings/:id/monthly?run_id=&model=
// The twelve monthly gaps for the scoring year, for the bar chart.
router.get("/buildings/:id/monthly", async (req, res, next) => {
  try {
    const { runId, model } = await params(req);
    const rows = await query(
      `SELECT month, actual_kwh, predicted_kwh, gap_kwh, gap_pct,
              z_score, coverage_pct
       FROM monthly_gaps
       WHERE run_id = $1 AND building_id = $2 AND model = $3
       ORDER BY month`,
      [runId, req.params.id, model]
    );
    res.json(rows);
  } catch (err) {
    next(err);
  }
});