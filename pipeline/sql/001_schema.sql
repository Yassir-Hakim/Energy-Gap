-- All tables for the energy gap tool.
-- Safe to re-run: it drops everything and recreates it from scratch.
-- Order matters: drop children before parents, create parents before children.

DROP TABLE IF EXISTS building_scores CASCADE;
DROP TABLE IF EXISTS monthly_gaps CASCADE;
DROP TABLE IF EXISTS daily_summary CASCADE;
DROP TABLE IF EXISTS building_models CASCADE;
DROP TABLE IF EXISTS model_runs CASCADE;
DROP TABLE IF EXISTS meter_readings CASCADE;
DROP TABLE IF EXISTS weather CASCADE;
DROP TABLE IF EXISTS buildings CASCADE;


-- One row per building. The reference table everything else points at.
CREATE TABLE buildings (
  building_id      text PRIMARY KEY,
  site_id          text NOT NULL,
  primary_use      text,
  sub_primary_use  text,
  sqm              double precision,   -- present for all 1,636 buildings
  sqft             double precision,
  timezone         text,
  year_built       integer,            -- blank for 819 buildings
  floors           integer,            -- blank for 1,195 buildings
  occupants        double precision,
  lat              double precision,
  lng              double precision,
  -- Stated heating fuel, filled in for 215 buildings (172 of them Gas).
  heating_type     text,
  -- Which meters this building has. Together with heating_type these tell us
  -- the heating fuel before any model is fitted: a building with a steam,
  -- hot-water or gas meter does not heat with electricity, so a clean result
  -- for it means "no electricity gap", not "no performance gap".
  has_electricity  boolean NOT NULL DEFAULT false,
  has_chilledwater boolean NOT NULL DEFAULT false,
  has_steam        boolean NOT NULL DEFAULT false,
  has_hotwater     boolean NOT NULL DEFAULT false,
  has_gas          boolean NOT NULL DEFAULT false
);


-- One row per site per hour. Shared by every building on that site.
-- 331,166 rows across 19 sites.
CREATE TABLE weather (
  site_id      text NOT NULL,
  ts           timestamp NOT NULL,     -- local site time, no time zone
  air_temp     double precision,
  dew_temp     double precision,
  cloud_cover  double precision,
  wind_speed   double precision,
  wind_dir     double precision,
  precip_1h    double precision,
  pressure     double precision,
  PRIMARY KEY (site_id, ts)
);


-- One row per building per hour. 25,212,579 rows.
-- No primary key here on purpose: it is added after loading ,
-- because building the index once at the end is far faster than maintaining
-- it while 25 million rows arrive.
-- kwh is double precision rather than real: summing a year of readings in
CREATE TABLE meter_readings (
  building_id  text NOT NULL,
  ts           timestamp NOT NULL,
  kwh          double precision NOT NULL
);


-- One row per pipeline run, so results can be compared between runs.
CREATE TABLE model_runs (
  run_id      serial PRIMARY KEY,      -- Postgres numbers these automatically
  created_at  timestamp NOT NULL DEFAULT now(),
  config      jsonb NOT NULL,          -- snapshot of config.py at run time
  notes       text
);


-- How well each model fitted the baseline year, for each building.
CREATE TABLE building_models (
  run_id          integer NOT NULL REFERENCES model_runs(run_id),
  building_id     text NOT NULL REFERENCES buildings(building_id),
  model           text NOT NULL,       -- 'towt' or 'gbm'
  n_train         integer,
  n_params        integer,
  cvrmse          double precision,
  nmbe            double precision,
  r2              double precision,
  cvrmse_holdout  double precision,    -- fit on held-out weeks: catches memorising
  nmbe_holdout    double precision,
  passes_g14      boolean,
  resid_sigma     double precision,    -- both feed the gap uncertainty
  resid_rho       double precision,
  params          jsonb,               -- TOWT coefficients; GBM hyperparameters
  PRIMARY KEY (run_id, building_id, model)
);


-- Daily actual vs predicted. This is what the charts read.
CREATE TABLE daily_summary (
  run_id         integer NOT NULL,
  building_id    text NOT NULL,
  model          text NOT NULL,
  day            date NOT NULL,
  period         text NOT NULL,        -- 'baseline' (2016) or 'scoring' (2017)
  actual_kwh     double precision,
  predicted_kwh  double precision,
  n_hours        integer,
  PRIMARY KEY (run_id, building_id, model, day)
);


-- The twelve monthly gaps for the scoring year.
CREATE TABLE monthly_gaps (
  run_id         integer NOT NULL,
  building_id    text NOT NULL,
  model          text NOT NULL,
  month          date NOT NULL,        -- first day of the month
  actual_kwh     double precision,
  predicted_kwh  double precision,
  gap_kwh        double precision,
  gap_pct        double precision,
  z_score        double precision,
  coverage_pct   double precision,
  PRIMARY KEY (run_id, building_id, model, month)
);


-- One headline verdict per building per model. The table the website ranks.
CREATE TABLE building_scores (
  run_id          integer NOT NULL,
  building_id     text NOT NULL,
  model           text NOT NULL,
  baseline_hours  integer,
  scoring_hours   integer,
  coverage_pct    double precision,
  actual_kwh      double precision,
  predicted_kwh   double precision,
  gap_kwh         double precision,
  gap_pct         double precision,
  z_score         double precision,
  -- over / under / ok / unreliable_model / insufficient_data
  status          text NOT NULL,
  PRIMARY KEY (run_id, building_id, model)
);