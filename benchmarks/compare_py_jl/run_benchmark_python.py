#!/usr/bin/env python3
"""
Full algorithm benchmark: Python DoubleML side.
Writes shared CSVs, fold indices, and timed results to data/benchmark_py.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.did.datasets import make_did_CS2021, make_did_SZ2020
from doubleml.irm.datasets import make_iivm_data, make_irm_data, make_ssm_data
from doubleml.plm.datasets import make_pliv_CHS2015, make_plpr_CP2025, make_plr_CCDDHNR2018
from doubleml.rdd.datasets import make_simple_rdd_data
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 3141
N_FOLDS = 5
# Warm multi-rep timing: 1 cold discard + median of N_WARM_REPS timed fits.
N_WARM_REPS = 3
N_COLD_DISCARD = 1


def ols():
    return LinearRegression()


def logit():
    return LogisticRegression(C=1e6, max_iter=500, solver="lbfgs")


def fixed_smpls(n: int, n_folds: int = N_FOLDS, seed: int = SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr.astype(int), te.astype(int)) for tr, te in kf.split(np.zeros(n))]


def save_smpls(path: Path, smpls):
    path.write_text(json.dumps([{"train": tr.tolist(), "test": te.tolist()} for tr, te in smpls]))


def timed(fn):
    """Single-shot wall clock (legacy helper)."""
    t0 = time.perf_counter()
    obj = fn()
    return obj, time.perf_counter() - t0


def timed_warm(fn, n_warm: int = N_WARM_REPS, n_cold: int = N_COLD_DISCARD):
    """Discard cold fit(s); return last object + median of warm reps + timing fields."""
    cold_secs = []
    obj = None
    for _ in range(n_cold):
        t0 = time.perf_counter()
        obj = fn()
        cold_secs.append(time.perf_counter() - t0)
    warm_secs = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        obj = fn()
        warm_secs.append(time.perf_counter() - t0)
    sec = float(np.median(warm_secs))
    meta = {
        "seconds_cold": float(cold_secs[0]) if cold_secs else None,
        "seconds_reps": [float(x) for x in warm_secs],
        "seconds_protocol": "warm_median",
        "n_warm_reps": int(n_warm),
        "n_cold_discard": int(n_cold),
    }
    return obj, sec, meta


def pack(est, seconds: float, theta_true=None, extra=None, timing=None):
    d = {
        "coef": np.asarray(est.coef, dtype=float).reshape(-1).tolist(),
        "se": np.asarray(est.se, dtype=float).reshape(-1).tolist(),
        "seconds": seconds,
        "n_coef": int(np.size(est.coef)),
    }
    if theta_true is not None:
        d["theta_true"] = theta_true if isinstance(theta_true, list) else float(theta_true)
    if timing:
        d.update(timing)
    if extra:
        d.update(extra)
    return d


def main():
    results = {
        "backend": "python",
        "doubleml": dml.__version__,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "timing_protocol": "warm_median",
        "n_warm_reps": N_WARM_REPS,
        "n_cold_discard": N_COLD_DISCARD,
        "models": {},
    }

    # ---- PLR ----
    np.random.seed(SEED)
    df = make_plr_CCDDHNR2018(n_obs=1000, dim_x=10, alpha=0.5, return_type="DataFrame")
    df.to_csv(OUT / "bench_plr.csv", index=False)
    smpls = fixed_smpls(len(df), seed=SEED)
    save_smpls(OUT / "bench_plr_smpls.json", smpls)
    data = dml.DoubleMLData(df, "y", "d")

    def fit_plr():
        m = dml.DoubleMLPLR(data, ols(), ols(), n_folds=N_FOLDS, draw_sample_splitting=False)
        m.set_sample_splitting(smpls)
        m.fit()
        return m

    est, sec, tmeta = timed_warm(fit_plr)
    results["models"]["PLR"] = pack(est, sec, 0.5, timing=tmeta)
    fw = est.construct_framework()
    results["models"]["Framework_2x_PLR"] = {
        "coef": (2 * fw).thetas.tolist(),
        "se": (2 * fw).ses.tolist(),
        "seconds": 0.0,
        "seconds_cold": 0.0,
        "seconds_reps": [],
        "seconds_protocol": "bookkeeping",
        "theta_true": 1.0,
        "note": "2 * construct_framework(PLR)",
        "exclude_from_runtime_total": True,
    }
    # sensitivity on same PLR fit
    est.sensitivity_analysis(cf_y=0.04, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0)
    sp = est.sensitivity_params
    results["models"]["PLR_sensitivity"] = {
        "coef": est.coef.tolist(),
        "se": est.se.tolist(),
        "theta_lower": np.asarray(sp["theta"]["lower"]).reshape(-1).tolist(),
        "theta_upper": np.asarray(sp["theta"]["upper"]).reshape(-1).tolist(),
        "ci_lower": np.asarray(sp["ci"]["lower"]).reshape(-1).tolist(),
        "ci_upper": np.asarray(sp["ci"]["upper"]).reshape(-1).tolist(),
        "rv": np.asarray(sp["rv"]).reshape(-1).tolist(),
        "rva": np.asarray(sp["rva"]).reshape(-1).tolist(),
        "seconds": 0.0,
        "seconds_cold": 0.0,
        "seconds_reps": [],
        "seconds_protocol": "bookkeeping",
        "cf_y": 0.04,
        "cf_d": 0.03,
        "rho": 1.0,
        "exclude_from_runtime_total": True,
    }

    # ---- IRM ----
    np.random.seed(SEED + 1)
    df = make_irm_data(n_obs=1200, dim_x=5, theta=0.5, return_type="DataFrame")
    df.to_csv(OUT / "bench_irm.csv", index=False)
    smpls = fixed_smpls(len(df), seed=SEED + 1)
    save_smpls(OUT / "bench_irm_smpls.json", smpls)
    data = dml.DoubleMLData(df, "y", "d")

    def fit_irm():
        m = dml.DoubleMLIRM(
            data, ols(), logit(), n_folds=N_FOLDS, draw_sample_splitting=False, trimming_threshold=0.01
        )
        m.set_sample_splitting(smpls)
        m.fit()
        return m

    est, sec, tmeta = timed_warm(fit_irm)
    results["models"]["IRM"] = pack(est, sec, 0.5, timing=tmeta)

    # ---- PLIV ----
    np.random.seed(SEED + 2)
    df = make_pliv_CHS2015(n_obs=1200, alpha=1.0, dim_x=10, dim_z=1, return_type="DataFrame")
    df.to_csv(OUT / "bench_pliv.csv", index=False)
    smpls = fixed_smpls(len(df), seed=SEED + 2)
    save_smpls(OUT / "bench_pliv_smpls.json", smpls)
    zcols = [c for c in df.columns if str(c).startswith("Z")]
    data = dml.DoubleMLData(df, "y", "d", z_cols=zcols)

    def fit_pliv():
        m = dml.DoubleMLPLIV(data, ols(), ols(), ols(), n_folds=N_FOLDS, draw_sample_splitting=False)
        m.set_sample_splitting(smpls)
        m.fit()
        return m

    est, sec, tmeta = timed_warm(fit_pliv)
    results["models"]["PLIV"] = pack(est, sec, 1.0, timing=tmeta)

    # ---- IIVM ----
    np.random.seed(SEED + 3)
    df = make_iivm_data(n_obs=2500, dim_x=5, theta=0.5, return_type="DataFrame")
    df.to_csv(OUT / "bench_iivm.csv", index=False)
    smpls = fixed_smpls(len(df), seed=SEED + 3)
    save_smpls(OUT / "bench_iivm_smpls.json", smpls)
    zcols = [c for c in df.columns if str(c).lower().startswith("z")]
    data = dml.DoubleMLData(df, "y", "d", z_cols=zcols)

    def fit_iivm():
        m = dml.DoubleMLIIVM(
            data, ols(), logit(), logit(), n_folds=N_FOLDS, draw_sample_splitting=False, trimming_threshold=0.05
        )
        m.set_sample_splitting(smpls)
        m.fit()
        return m

    est, sec, tmeta = timed_warm(fit_iivm)
    results["models"]["IIVM"] = pack(est, sec, 0.5, timing=tmeta)

    # ---- multi-treatment PLR ----
    np.random.seed(SEED + 4)
    n, p = 1200, 6
    X = np.random.randn(n, p)
    b = X[:, 0] + 0.25 * X[:, 1] ** 2
    D = np.column_stack([0.4 * X[:, 0] + np.random.randn(n), 0.4 * X[:, 1] + np.random.randn(n)])
    theta = np.array([0.5, -0.3])
    y = D @ theta + b + np.random.randn(n)
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
    df["y"] = y
    df["d1"] = D[:, 0]
    df["d2"] = D[:, 1]
    df.to_csv(OUT / "bench_plr_multi.csv", index=False)
    smpls = fixed_smpls(n, seed=SEED + 4)
    save_smpls(OUT / "bench_plr_multi_smpls.json", smpls)
    data = dml.DoubleMLData(df, "y", ["d1", "d2"])

    def fit_multi():
        m = dml.DoubleMLPLR(data, ols(), ols(), n_folds=N_FOLDS, draw_sample_splitting=False)
        m.set_sample_splitting(smpls)
        m.fit()
        return m

    est, sec, tmeta = timed_warm(fit_multi)
    results["models"]["PLR_multi"] = pack(est, sec, theta.tolist(), timing=tmeta)

    # ---- PLPR all approaches ----
    np.random.seed(SEED + 5)
    df = make_plpr_CP2025(num_id=150, num_t=5, dim_x=6, theta=0.5, dgp_type="dgp1")
    df.to_csv(OUT / "bench_plpr.csv", index=False)
    for approach in ("fd_exact", "wg_approx", "cre_general", "cre_normal"):
        def fit_plpr(ap=approach):
            panel = dml.DoubleMLPanelData(df, y_col="y", d_cols="d", t_col="time", id_col="id", static_panel=True)
            m = dml.DoubleMLPLPR(panel, ols(), ols(), approach=ap, n_folds=3)
            m.fit()
            return m

        est, sec, tmeta = timed_warm(fit_plpr)
        results["models"][f"PLPR_{approach}"] = pack(est, sec, 0.5, {"n_folds": 3}, timing=tmeta)

    # ---- DID two-period ----
    np.random.seed(SEED + 6)
    df = make_did_SZ2020(n_obs=1000, dgp_type=1, cross_sectional_data=False, return_type="DataFrame")
    if "y" not in df.columns and {"y0", "y1"}.issubset(df.columns):
        df = df.copy()
        df["y"] = df["y1"] - df["y0"]
    df.to_csv(OUT / "bench_did.csv", index=False)
    smpls = fixed_smpls(len(df), seed=SEED + 6)
    save_smpls(OUT / "bench_did_smpls.json", smpls)
    xcols = [c for c in df.columns if str(c).startswith("X") or str(c).startswith("Z") or str(c).startswith("x")]
    data = dml.DoubleMLDIDData(df, "y", "d", x_cols=xcols if xcols else None)

    def fit_did():
        m = dml.DoubleMLDID(
            data, ols(), logit(), n_folds=N_FOLDS, draw_sample_splitting=False, score="observational"
        )
        m.set_sample_splitting(smpls)
        m.fit()
        return m

    try:
        est, sec, tmeta = timed_warm(fit_did)
        results["models"]["DID"] = pack(est, sec, timing=tmeta)
    except Exception as e:
        results["models"]["DID"] = {"error": str(e), "seconds": None}

    # ---- DID multi: CS2021 → integer t; never-treated d = +inf for Python float panels ----
    np.random.seed(SEED + 7)
    try:
        df0 = make_did_CS2021(n_obs=150, dgp_type=1, return_type="DataFrame")
        t_levels = sorted(df0["t"].dropna().unique())
        t_map = {t: i + 1 for i, t in enumerate(t_levels)}
        # Julia CSV uses 0 for never-treated; Python float panel expects +inf
        d_jl = df0["d"].map(lambda x: 0.0 if pd.isna(x) else float(t_map[x]))
        d_py = df0["d"].map(lambda x: np.inf if pd.isna(x) else float(t_map[x]))
        df_jl = pd.DataFrame({
            "id": df0["id"].astype(int),
            "t": df0["t"].map(t_map).astype(int),
            "y": df0["y"].astype(float),
            "d": d_jl.astype(float),
            "X1": df0["Z1"], "X2": df0["Z2"], "X3": df0["Z3"], "X4": df0["Z4"],
        })
        df_jl.to_csv(OUT / "bench_did_multi.csv", index=False)
        df_py = df_jl.copy()
        df_py["d"] = d_py.astype(float)
        panel = dml.DoubleMLPanelData(df_py, y_col="y", d_cols="d", t_col="t", id_col="id")

        def fit_didm():
            m = dml.did.DoubleMLDIDMulti(
                panel, ols(), logit(), n_folds=3, control_group="never_treated",
                gt_combinations="standard",
            )
            m.fit()
            return m

        est, sec, tmeta = timed_warm(fit_didm)
        results["models"]["DID_multi"] = pack(
            est, sec,
            extra={"n_att": int(np.size(est.coef)), "coef_mean": float(np.mean(est.coef))},
            timing=tmeta,
        )
    except Exception as e:
        results["models"]["DID_multi"] = {"error": str(e), "seconds": None}

    # ---- RDFlex ----
    np.random.seed(SEED + 8)
    try:
        dct = make_simple_rdd_data(n_obs=2000, fuzzy=False)
        df = pd.DataFrame(dct["X"], columns=[f"X{i+1}" for i in range(dct["X"].shape[1])])
        df["y"] = dct["Y"]
        df["d"] = dct["D"]
        df["score"] = dct["score"]
        df.to_csv(OUT / "bench_rdd.csv", index=False)
        obj = dml.DoubleMLRDDData.from_arrays(x=dct["X"], y=dct["Y"], d=dct["D"], score=dct["score"])

        def fit_rdd():
            m = dml.rdd.RDFlex(obj, ols(), fuzzy=False, n_folds=3, n_rep=1, fs_specification="cutoff")
            m.fit(n_iterations=2)
            return m

        est, sec, tmeta = timed_warm(fit_rdd)
        # rdrobust returns Conventional / Bias-Corrected / Robust — keep Conventional only for parity
        coef = np.asarray(est.coef, dtype=float).reshape(-1)
        se = np.asarray(est.se, dtype=float).reshape(-1)
        results["models"]["RDFlex"] = {
            "coef": [float(coef[0])],
            "se": [float(se[0])],
            "seconds": sec,
            "n_coef": 1,
            "n_iterations": 2,
            "note": "Python Conventional estimate (rdrobust)",
            **tmeta,
        }
    except Exception as e:
        results["models"]["RDFlex"] = {"error": str(e), "seconds": None}

    # ---- SSM MAR ----
    np.random.seed(SEED + 9)
    try:
        df = make_ssm_data(n_obs=1500, dim_x=4, theta=1.0, mar=True, return_type="DataFrame")
        df.to_csv(OUT / "bench_ssm_mar.csv", index=False)
        smpls = fixed_smpls(len(df), seed=SEED + 9)
        save_smpls(OUT / "bench_ssm_mar_smpls.json", smpls)
        data = dml.DoubleMLSSMData(df, y_col="y", d_cols="d", s_col="s")

        def fit_ssm_mar():
            m = dml.DoubleMLSSM(
                data, ols(), logit(), logit(),
                n_folds=N_FOLDS, draw_sample_splitting=False, score="missing-at-random",
                trimming_threshold=0.05,
            )
            m.set_sample_splitting(smpls)
            m.fit()
            return m

        est, sec, tmeta = timed_warm(fit_ssm_mar)
        results["models"]["SSM_MAR"] = pack(est, sec, 1.0, timing=tmeta)
    except Exception as e:
        results["models"]["SSM_MAR"] = {"error": str(e), "seconds": None}

    # ---- SSM nonignorable ----
    np.random.seed(SEED + 10)
    try:
        df = make_ssm_data(n_obs=1500, dim_x=4, theta=1.0, mar=False, return_type="DataFrame")
        df.to_csv(OUT / "bench_ssm_ni.csv", index=False)
        smpls = fixed_smpls(len(df), seed=SEED + 10)
        save_smpls(OUT / "bench_ssm_ni_smpls.json", smpls)
        zcols = [c for c in df.columns if str(c).lower().startswith("z")]
        data = dml.DoubleMLSSMData(df, y_col="y", d_cols="d", s_col="s", z_cols=zcols)

        def fit_ssm_ni():
            m = dml.DoubleMLSSM(
                data, ols(), logit(), logit(),
                n_folds=N_FOLDS, draw_sample_splitting=False, score="nonignorable",
                trimming_threshold=0.05,
            )
            m.set_sample_splitting(smpls)
            m.fit()
            return m

        est, sec, tmeta = timed_warm(fit_ssm_ni)
        results["models"]["SSM_nonignorable"] = pack(est, sec, 1.0, timing=tmeta)
    except Exception as e:
        results["models"]["SSM_nonignorable"] = {"error": str(e), "seconds": None}

    out = OUT / "benchmark_py.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
