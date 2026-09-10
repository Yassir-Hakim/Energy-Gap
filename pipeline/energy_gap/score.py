import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from energy_gap import config
from energy_gap.db import copy_frame, execute, get_conn, query_df
from energy_gap.features import build_model_frame, holdout_mask
from energy_gap.metrics import cvrmse, nmbe, r2, resid_stats
from energy_gap.models import fit_gbm, fit_towt

# The one module in v0.2 that touches the database. Everything it calls in
# between - features, models, metrics - is pure computation on DataFrames.
#
# Per building: read, fit, measure, score, write. One building at a time,
# because 25 million rows will not sit in memory alongside a model fit, and
# because one building failing should not take down the run.


def _building_list(limit=None, sites=None):
    # Counted outward from buildings, NOT inward from meter_readings.
    #
    # This matters. 46 buildings carry an electricity flag but have no 2016
    # rows at all, so grouping meter_readings produces no row for them and a
    # HAVING clause never sees them. They must be scored insufficient_data,
    # not silently skipped. data_checks.py counts the same way.
    sql = "SELECT building_id, site_id FROM buildings WHERE has_electricity"
    params = []
    if sites:
        sql += " AND site_id = ANY(%s)"
        params.append(list(sites))
    sql += " ORDER BY building_id"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    return query_df(sql, params or None)


def _read_building(building_id, site_id):
    # %s placeholders
    readings = query_df(
        "SELECT ts, kwh FROM meter_readings WHERE building_id = %s ORDER BY ts",
        (building_id,),
    )
    weather = query_df(
        "SELECT ts, air_temp, dew_temp, cloud_cover, wind_speed "
        "FROM weather WHERE site_id = %s ORDER BY ts",
        (site_id,),
    )
    return readings, weather


def _fit_stats(model, frame, holdout_cv=None, holdout_nmbe=None):
    # Everything that goes in one building_models row.
    pred = model.predict(frame)
    y = frame["kwh"].to_numpy(dtype=float)

    cv = cvrmse(y, pred, model.n_params)
    nb = nmbe(y, pred, model.n_params)
    sigma, rho = resid_stats(y, pred)

    # The Guideline 14 gate. A building that fails this is not reported, no
    # matter how large its gap: the gap would be measuring model error rather
    # than building behaviour.
    passes = (cv <= config.G14_HOURLY_CVRMSE_MAX
              and abs(nb) <= config.G14_HOURLY_NMBE_ABS_MAX)

    return {
        "n_train": len(frame),
        "n_params": model.n_params,
        "cvrmse": cv,
        "nmbe": nb,
        "r2": r2(y, pred),
        "cvrmse_holdout": holdout_cv,
        "nmbe_holdout": holdout_nmbe,
        "passes_g14": passes,
        "resid_sigma": sigma,
        "resid_rho": rho,
        "params": json.dumps(model.params()),
    }


def _holdout_stats(fit_fn, frame):
    # Fit on the held-in weeks only, then measure on weeks the model never saw.
    #
    # Why this exists: a model with 175 parameters can fit noise. If it scores
    # well on hours it trained on but badly on unseen weeks, it memorised
    # rather than learned. Comparing in-sample against holdout is the only way
    # to tell the difference.
    #
    # Whole weeks rather than random hours, because consecutive hours are
    # highly correlated - holding out 3pm while training on 2pm and 4pm is
    # barely a test.
    held = holdout_mask(frame["ts"])
    if held.sum() < 100 or (~held).sum() < 500:
        return None, None, None, None

    train = frame[~held].reset_index(drop=True)
    test = frame[held].reset_index(drop=True)

    model = fit_fn(train)
    pred = model.predict(test)
    y = test["kwh"].to_numpy(dtype=float)

    # sigma and rho come from the HELD-OUT residuals, not the in-sample ones.
    #
    # Predicting the scoring year is an out-of-sample task, so the uncertainty
    # attached to it should be estimated out of sample too. In-sample residuals
    # measure how well a model memorised, which is a different thing.
    #
    # This matters most for a model that overfits: its in-sample sigma is
    # understated, and sigma sits in the z-score denominator, so an overfitted
    # model's gaps would come out looking MORE significant than an honest
    # model's. Check the holdout column in summarise() to see whether either
    # model is doing this.
    sigma, rho = resid_stats(y, pred)
    return cvrmse(y, pred, model.n_params), nmbe(y, pred, model.n_params), sigma, rho


def _gap_uncertainty(gap_kwh, n_hours, sigma, rho):
    # Standard error of a summed gap, and the z-score that follows.
    #
    # The (1+rho)/(1-rho) factor corrects for autocorrelation. Residuals in
    # consecutive hours are highly correlated - if a building is running hot at
    # 2pm it is probably running hot at 3pm - so summing m of them does NOT
    # average the error away the way m independent draws would. Without the
    # correction every gap looks far more significant than it is. Guideline 14
    # uses the same idea in its savings-uncertainty method.
    if sigma is None or n_hours <= 0:
        return None, None
    rho = 0.0 if rho is None or not np.isfinite(rho) else min(max(rho, -0.99), 0.99)
    se = sigma * np.sqrt(n_hours * (1 + rho) / (1 - rho))
    if se <= 0:
        return None, None
    return se, gap_kwh / se


def _status(baseline_hours, coverage_pct, passes_g14, gap_pct, z):
    # Order matters: data quality first, then model quality, then the verdict.
    # A building only gets flagged if its own baseline model earned the right
    # to be believed.
    if baseline_hours < config.MIN_BASELINE_HOURS:
        return "insufficient_data"
    if coverage_pct is None or coverage_pct < config.MIN_COVERAGE_PCT:
        return "insufficient_data"
    if not passes_g14:
        return "unreliable_model"
    if gap_pct is None or z is None:
        return "ok"
    # Two independent tests: the gap must be big enough to matter (materiality)
    # and clear enough of the noise to be believed (evidence).
    if gap_pct >= config.GAP_FLAG_PCT and z >= config.Z_FLAG:
        return "over"
    if gap_pct <= -config.GAP_FLAG_PCT and z <= -config.Z_FLAG:
        return "under"
    return "ok"


def _periods(scoring, model, sigma, rho):
    # Daily rows for the charts, monthly rows for the bar chart, and the annual
    # totals that decide the status.
    pred = model.predict(scoring)
    df = pd.DataFrame({
        "ts": pd.to_datetime(scoring["ts"]),
        "actual": scoring["kwh"].to_numpy(dtype=float),
        "predicted": pred,
    })

    daily = df.groupby(df["ts"].dt.date).agg(
        actual_kwh=("actual", "sum"),
        predicted_kwh=("predicted", "sum"),
        n_hours=("actual", "size"),
    ).reset_index(names="day")

    monthly = df.groupby(df["ts"].dt.to_period("M")).agg(
        actual_kwh=("actual", "sum"),
        predicted_kwh=("predicted", "sum"),
        n_hours=("actual", "size"),
    ).reset_index(names="month")
    monthly["month"] = monthly["month"].dt.to_timestamp().dt.date

    monthly["gap_kwh"] = monthly["actual_kwh"] - monthly["predicted_kwh"]
    monthly["gap_pct"] = 100 * monthly["gap_kwh"] / monthly["predicted_kwh"]
    monthly["z_score"] = [
        _gap_uncertainty(g, n, sigma, rho)[1]
        for g, n in zip(monthly["gap_kwh"], monthly["n_hours"])
    ]
    # Hours in each month of the scoring year, for the coverage percentage.
    hours_in_month = pd.to_datetime(monthly["month"]).dt.days_in_month * 24
    monthly["coverage_pct"] = 100 * monthly["n_hours"] / hours_in_month

    return daily, monthly


def _score_one(building_id, site_id, run_id):
    # Returns four lists of rows, one per destination table. Empty lists are
    # fine; a building with no usable data still gets a building_scores row so
    # it can be counted rather than vanishing.
    readings, weather = _read_building(building_id, site_id)

    baseline = build_model_frame(readings, weather, year=config.BASELINE_YEAR)
    scoring = build_model_frame(readings, weather, year=config.SCORING_YEAR)

    models_rows, daily_rows, monthly_rows, score_rows = [], [], [], []

    # Not enough baseline data to fit anything. Recorded, not skipped.
    if len(baseline) < config.MIN_BASELINE_HOURS:
        for name in ("towt", "gbm"):
            score_rows.append({
                "run_id": run_id, "building_id": building_id, "model": name,
                "baseline_hours": len(baseline), "scoring_hours": len(scoring),
                "coverage_pct": None, "actual_kwh": None, "predicted_kwh": None,
                "gap_kwh": None, "gap_pct": None, "z_score": None,
                "status": "insufficient_data",
            })
        return models_rows, daily_rows, monthly_rows, score_rows

    scoring_hours_possible = 8760 if config.SCORING_YEAR % 4 else 8784
    coverage = 100 * len(scoring) / scoring_hours_possible

    for name, fit_fn in (("towt", fit_towt), ("gbm", fit_gbm)):
        # The model that scores 2017 is fitted on the FULL baseline year. The
        # holdout model is fitted separately, measured, and discarded - it
        # exists to produce an honesty statistic, not to withhold data from the
        # model actually used.
        model = fit_fn(baseline)
        hcv, hnb, hsigma, hrho = _holdout_stats(fit_fn, baseline)
        stats = _fit_stats(model, baseline, hcv, hnb)

        # Prefer the out-of-sample residual spread for uncertainty; fall back to
        # in-sample only when the building had too little data to hold weeks out.
        unc_sigma = hsigma if hsigma is not None else stats["resid_sigma"]
        unc_rho = hrho if hrho is not None else stats["resid_rho"]

        models_rows.append({
            "run_id": run_id, "building_id": building_id, "model": name, **stats
        })

        # Baseline-year daily rows, so the chart can show both periods.
        base_pred = model.predict(baseline)
        base_df = pd.DataFrame({
            "ts": pd.to_datetime(baseline["ts"]),
            "actual": baseline["kwh"].to_numpy(dtype=float),
            "predicted": base_pred,
        })
        base_daily = base_df.groupby(base_df["ts"].dt.date).agg(
            actual_kwh=("actual", "sum"),
            predicted_kwh=("predicted", "sum"),
            n_hours=("actual", "size"),
        ).reset_index(names="day")
        for _, d in base_daily.iterrows():
            daily_rows.append({
                "run_id": run_id, "building_id": building_id, "model": name,
                "day": d["day"], "period": "baseline",
                "actual_kwh": d["actual_kwh"], "predicted_kwh": d["predicted_kwh"],
                "n_hours": int(d["n_hours"]),
            })

        if scoring.empty:
            score_rows.append({
                "run_id": run_id, "building_id": building_id, "model": name,
                "baseline_hours": len(baseline), "scoring_hours": 0,
                "coverage_pct": 0.0, "actual_kwh": None, "predicted_kwh": None,
                "gap_kwh": None, "gap_pct": None, "z_score": None,
                "status": "insufficient_data",
            })
            continue

        daily, monthly = _periods(scoring, model, unc_sigma, unc_rho)

        for _, d in daily.iterrows():
            daily_rows.append({
                "run_id": run_id, "building_id": building_id, "model": name,
                "day": d["day"], "period": "scoring",
                "actual_kwh": d["actual_kwh"], "predicted_kwh": d["predicted_kwh"],
                "n_hours": int(d["n_hours"]),
            })

        for _, m in monthly.iterrows():
            monthly_rows.append({
                "run_id": run_id, "building_id": building_id, "model": name,
                "month": m["month"], "actual_kwh": m["actual_kwh"],
                "predicted_kwh": m["predicted_kwh"], "gap_kwh": m["gap_kwh"],
                "gap_pct": m["gap_pct"], "z_score": m["z_score"],
                "coverage_pct": m["coverage_pct"],
            })

        actual = float(monthly["actual_kwh"].sum())
        predicted = float(monthly["predicted_kwh"].sum())
        gap_kwh = actual - predicted
        gap_pct = 100 * gap_kwh / predicted if predicted else None
        _, z = _gap_uncertainty(gap_kwh, len(scoring), unc_sigma, unc_rho)

        score_rows.append({
            "run_id": run_id, "building_id": building_id, "model": name,
            "baseline_hours": len(baseline), "scoring_hours": len(scoring),
            "coverage_pct": coverage, "actual_kwh": actual,
            "predicted_kwh": predicted, "gap_kwh": gap_kwh, "gap_pct": gap_pct,
            "z_score": z,
            "status": _status(len(baseline), coverage, stats["passes_g14"], gap_pct, z),
        })

    return models_rows, daily_rows, monthly_rows, score_rows


def _new_run(notes=None):
    # A snapshot of config.py, so a set of results can always be traced back to
    # the settings that produced them.
    snapshot = {
        "baseline_year": config.BASELINE_YEAR,
        "scoring_year": config.SCORING_YEAR,
        "temp_knots_c": config.TEMP_KNOTS_C,
        "weather_ffill_limit_hours": config.WEATHER_FFILL_LIMIT_HOURS,
        "min_baseline_hours": config.MIN_BASELINE_HOURS,
        "min_coverage_pct": config.MIN_COVERAGE_PCT,
        "g14_cvrmse_max": config.G14_HOURLY_CVRMSE_MAX,
        "g14_nmbe_abs_max": config.G14_HOURLY_NMBE_ABS_MAX,
        "gap_flag_pct": config.GAP_FLAG_PCT,
        "z_flag": config.Z_FLAG,
        "holdout_every_nth_week": config.HOLDOUT_EVERY_NTH_WEEK,
        "ridge_alpha": config.RIDGE_ALPHA,
        "gbm_params": config.GBM_PARAMS,
    }
    rows = execute(
        "INSERT INTO model_runs (config, notes) VALUES (%s, %s) RETURNING run_id",
        (json.dumps(snapshot), notes),
    )
    return rows[0][0]


def _write(models_rows, daily_rows, monthly_rows, score_rows):
    for rows, table in (
        (models_rows, "building_models"),
        (daily_rows, "daily_summary"),
        (monthly_rows, "monthly_gaps"),
        (score_rows, "building_scores"),
    ):
        if rows:
            df = pd.DataFrame(rows)
            copy_frame(df, table, list(df.columns))


def _score_one_safe(building_id, site_id, run_id):
    # Runs inside a worker process. Exceptions are caught and returned rather
    # than raised, because an exception crossing a process boundary would take
    # down the whole batch instead of one building.
    try:
        return _score_one(building_id, site_id, run_id), None
    except Exception as e:
        return ([], [], [], []), (building_id, f"{type(e).__name__}: {e}")


def run(limit=None, sites=None, notes=None, n_jobs=-1, batch_size=50):
    """Score every building. n_jobs=-1 uses all cores, n_jobs=1 stays serial.

    Buildings are scored in parallel because each one is completely
    independent: its model is fitted on its own history and never sees another
    building's data. That independence is a property of the method, not a
    convenience, which is why this parallelises with no change to the results.

    Work is done in batches so that results are written as the run progresses
    rather than all at the end, which keeps memory flat and means a run
    interrupted halfway still leaves usable rows behind.

    Each worker opens its own database connection. psycopg connections cannot
    be shared across processes, and get_conn() is called inside _score_one, so
    this happens naturally.
    """
    buildings = _building_list(limit=limit, sites=sites)
    run_id = _new_run(notes)

    workers = os.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Run {run_id}: scoring {len(buildings):,} buildings on {workers} core(s)")

    failures = []

    for start in range(0, len(buildings), batch_size):
        chunk = buildings.iloc[start:start + batch_size]

        # loky is joblib's default backend: separate processes, so each gets
        # its own Python interpreter and its own database connection. It also
        # pins each worker to one thread, which stops scikit-learn's internal
        # threading from fighting the outer parallelism for cores.
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_score_one_safe)(b.building_id, b.site_id, run_id)
            for b in chunk.itertuples()
        )

        batch = ([], [], [], [])
        for rows, failure in results:
            if failure:
                failures.append(failure)
            for dest, src in zip(batch, rows):
                dest.extend(src)

        # Writing happens here, in the parent, not in the workers. Serialising
        # the writes avoids many processes contending for connections, and the
        # writes are a small fraction of the total time.
        _write(*batch)
        print(f"  {min(start + batch_size, len(buildings)):,}/{len(buildings):,} buildings")

    if failures:
        print(f"\n  {len(failures)} building(s) failed:")
        for bid, msg in failures[:10]:
            print(f"    {bid}: {msg}")

    return run_id


def summarise(run_id):
    print("\nStatus counts")
    print("-" * 52)
    counts = query_df("""
        SELECT model, status, count(*) AS n
        FROM building_scores WHERE run_id = %s
        GROUP BY model, status ORDER BY model, n DESC
    """, (run_id,))
    for model in counts["model"].unique():
        sub = counts[counts["model"] == model]
        print(f"  {model}")
        for _, r in sub.iterrows():
            print(f"    {r['status']:<20} {r['n']:>6,}")

    held = 52 // config.HOLDOUT_EVERY_NTH_WEEK
    kept = 52 - held

    print("\nFit quality")
    print("-" * 66)
    print("  Guideline 14 asks for CV(RMSE) <= 30% and |NMBE| <= 10% hourly.")
    print()
    print("  TWO SEPARATE MODELS are fitted per building. The first is the one")
    print("  that goes on to score the scoring year. The second is fitted,")
    print("  measured and thrown away; it exists only to check whether the")
    print("  first is learning the building or memorising its data.")
    print()
    fit = query_df("""
        SELECT model,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY cvrmse) AS cv_median,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY cvrmse_holdout) AS cv_holdout,
               100.0 * avg(CASE WHEN passes_g14 THEN 1 ELSE 0 END) AS pass_pct,
               -- The same G14 thresholds applied to the held-out weeks. Not
               -- part of the standard, which is in-sample only, but it is how
               -- ML-based M&V evaluates a baseline model: on data it did not
               -- see. Reported alongside, never gated on.
               100.0 * avg(CASE WHEN cvrmse_holdout <= %s
                            AND abs(nmbe_holdout) <= %s THEN 1 ELSE 0 END) AS pass_pct_holdout
        FROM building_models WHERE run_id = %s
        GROUP BY model ORDER BY model
    """, (config.G14_HOURLY_CVRMSE_MAX, config.G14_HOURLY_NMBE_ABS_MAX, run_id))

    for _, r in fit.iterrows():
        print(f"  {r['model']}")
        print(f"    model 1: trained on all 52 weeks, measured on those same 52"
              f"    Median CV(RMSE) {r['cv_median']:5.1f}%   G14 pass {r['pass_pct']:5.1f}%")
        print(f"    model 2: trained on {kept} weeks, measured on the {held} it never saw"
              f"    Median CV(RMSE) {r['cv_holdout']:5.1f}%   G14 pass {r['pass_pct_holdout']:5.1f}%")
    print()
    print("  Model 1's numbers come from hours it was fitted on, so they say how")
    print("  well it FITS. Model 2's come from weeks it never saw, so they say")
    print("  how well it PREDICTS. Both medians are across buildings.")
    print()
    print("  A large drop from line 1 to line 2 means the extra accuracy on")
    print("  line 1 is noise being fitted, and the G14 pass rate that gates")
    print("  unreliable_model is flattering the model.")

    print("\nModel agreement on over-consumption")
    print("-" * 52)
    agree = query_df("""
        SELECT count(*) FILTER (WHERE t.status = 'over' AND g.status = 'over') AS both,
               count(*) FILTER (WHERE t.status = 'over' AND g.status <> 'over') AS towt_only,
               count(*) FILTER (WHERE g.status = 'over' AND t.status <> 'over') AS gbm_only
        FROM building_scores t
        JOIN building_scores g
          ON t.building_id = g.building_id AND t.run_id = g.run_id
        WHERE t.run_id = %s AND t.model = 'towt' AND g.model = 'gbm'
    """, (run_id,)).iloc[0]
    print(f"  both models      {agree['both']:>6,}   <- highest confidence")
    print(f"  TOWT only        {agree['towt_only']:>6,}")
    print(f"  GBM only         {agree['gbm_only']:>6,}   <- check for hot-month extrapolation")

    print("\nLargest gaps flagged by both models")
    print("-" * 52)
    top = query_df("""
        SELECT t.building_id, b.primary_use, t.gap_pct, t.z_score, t.cvrmse
        FROM (SELECT s.*, m.cvrmse FROM building_scores s
              JOIN building_models m USING (run_id, building_id, model)
              WHERE s.run_id = %s AND s.model = 'towt') t
        JOIN building_scores g
          ON g.building_id = t.building_id AND g.run_id = %s AND g.model = 'gbm'
        JOIN buildings b ON b.building_id = t.building_id
        WHERE t.status = 'over' AND g.status = 'over'
        ORDER BY t.gap_pct DESC LIMIT 10
    """, (run_id, run_id))
    if top.empty:
        print("  none")
    else:
        for _, r in top.iterrows():
            use = (r["primary_use"] or "")[:22]
            print(f"  {r['building_id']:<28} {use:<24} "
                  f"{r['gap_pct']:+6.1f}%  z={r['z_score']:6.1f}  CV={r['cvrmse']:4.1f}%")
    print()

