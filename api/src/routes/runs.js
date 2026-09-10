// Two endpoints about the API itself and about pipeline runs.

import express from "express";
import { query } from "../db.js";

export const router = express.Router();

// GET /api/health
// Is the server up, and can it reach the database? The first thing to check
// when anything else fails, because it separates "API is down" from "API is
// up but the database is not".
router.get("/health", async (req, res) => {
  try {
    await query("SELECT 1");
    res.json({ ok: true, db: true });
  } catch {
    res.status(500).json({ ok: true, db: false });
  }
});

// GET /api/runs
// Every pipeline run, newest first. Each carries the config it ran with, so
// two sets of results can always be told apart.
router.get("/runs", async (req, res, next) => {
  try {
    const rows = await query(`
      SELECT r.run_id,
             r.created_at,
             r.notes,
             r.config,
             count(DISTINCT s.building_id) AS n_buildings
      FROM model_runs r
      LEFT JOIN building_scores s ON s.run_id = r.run_id
      GROUP BY r.run_id, r.created_at, r.notes, r.config
      ORDER BY r.run_id DESC
    `);
    res.json(rows);
  } catch (err) {
    next(err);
  }
});