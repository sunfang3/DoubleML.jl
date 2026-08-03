#!/usr/bin/env python3
"""Generate shared data + fixed folds, fit Python DoubleML, write results."""
from __future__ import annotations

import json
from pathlib import Path

import doubleml as dml
import numpy as np
import pandas as pd
from doubleml.irm.datasets import make_iivm_data, make_irm_data
from doubleml.plm.datasets import make_pliv_CHS2015, make_plr_CCDDHNR2018
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 3141
N_FOLDS = 5


def fixed_smpls(n: int, n_folds: int = N_FOLDS, seed: int = SEED):
    """Return DoubleML-style smpls: list of (train, test) for one rep."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train, test in kf.split(np.zeros(n)):
        folds.append((train.astype(int), test.astype(int)))
    return folds


def save_smpls(path: Path, smpls):
    payload = [{"train": tr.tolist(), "test": te.tolist()} for tr, te in smpls]
    path.write_text(json.dumps(payload))


def load_style_summary(est) -> dict:
    return {
        "coef": est.coef.tolist(),
        "se": est.se.tolist(),
        "t": est.t_stat.tolist(),
        "pval": est.pval.tolist(),
        "names": list(est.summary.index.astype(str)),
    }


def ols():
    return LinearRegression()


def logit():
    # nearly unregularized, comparable to Julia LogisticRegressionLearner(α small)
    return LogisticRegression(penalty="l2", C=1e6, max_iter=500, solver="lbfgs")


def main():
    results = {"python_doubleml": dml.__version__, "seed": SEED, "n_folds": N_FOLDS}

    # ----- PLR -----
    np.random.seed(SEED)
    df = make_plr_CCDDHNR2018(n_obs=800, dim_x=10, alpha=0.5, return_type="DataFrame")
    df.to_csv(OUT / "plr.csv", index=False)
    n = len(df)
    smpls = fixed_smpls(n)
    save_smpls(OUT / "plr_smpls.json", smpls)
    data = dml.DoubleMLData(df, "y", "d")
    plr = dml.DoubleMLPLR(data, ols(), ols(), n_folds=N_FOLDS, draw_sample_splitting=False)
    plr.set_sample_splitting(smpls)
    plr.fit()
    results["PLR"] = load_style_summary(plr)
    results["PLR"]["theta_true"] = 0.5

    # framework from PLR
    fw = plr.construct_framework()
    results["PLR_framework"] = {
        "thetas": fw.thetas.tolist(),
        "ses": fw.ses.tolist(),
    }
    # scale
    fw2 = 2 * fw
    results["PLR_framework_2x"] = {
        "thetas": fw2.thetas.tolist(),
        "ses": fw2.ses.tolist(),
    }

    # ----- IRM -----
    np.random.seed(SEED + 1)
    df = make_irm_data(n_obs=1000, dim_x=5, theta=0.5, return_type="DataFrame")
    # column names may be X1.. or 0..
    if "y" not in df.columns:
        # older API
        pass
    df.to_csv(OUT / "irm.csv", index=False)
    n = len(df)
    smpls = fixed_smpls(n, seed=SEED + 1)
    save_smpls(OUT / "irm_smpls.json", smpls)
    # detect d col
    dcol = "d" if "d" in df.columns else df.columns[-1]
    ycol = "y" if "y" in df.columns else [c for c in df.columns if c not in ("d",) and not str(c).startswith("X")][0]
    xcols = [c for c in df.columns if c not in (ycol, dcol)]
    data = dml.DoubleMLData(df, ycol, dcol, x_cols=xcols)
    irm = dml.DoubleMLIRM(data, ols(), logit(), n_folds=N_FOLDS, draw_sample_splitting=False, trimming_threshold=0.01)
    irm.set_sample_splitting(smpls)
    irm.fit()
    results["IRM"] = load_style_summary(irm)
    results["IRM"]["theta_true"] = 0.5

    # ----- PLIV -----
    np.random.seed(SEED + 2)
    df = make_pliv_CHS2015(n_obs=1000, alpha=1.0, dim_x=10, dim_z=1, return_type="DataFrame")
    df.to_csv(OUT / "pliv.csv", index=False)
    n = len(df)
    smpls = fixed_smpls(n, seed=SEED + 2)
    save_smpls(OUT / "pliv_smpls.json", smpls)
    zcols = [c for c in df.columns if str(c).startswith("Z")]
    data = dml.DoubleMLData(df, "y", "d", z_cols=zcols)
    pliv = dml.DoubleMLPLIV(data, ols(), ols(), ols(), n_folds=N_FOLDS, draw_sample_splitting=False)
    pliv.set_sample_splitting(smpls)
    pliv.fit()
    results["PLIV"] = load_style_summary(pliv)
    results["PLIV"]["theta_true"] = 1.0

    # ----- IIVM -----
    np.random.seed(SEED + 3)
    df = make_iivm_data(n_obs=2000, dim_x=5, theta=0.5, return_type="DataFrame")
    df.to_csv(OUT / "iivm.csv", index=False)
    n = len(df)
    smpls = fixed_smpls(n, seed=SEED + 3)
    save_smpls(OUT / "iivm_smpls.json", smpls)
    zcols = [c for c in df.columns if str(c).lower().startswith("z")]
    data = dml.DoubleMLData(df, "y", "d", z_cols=zcols)
    iivm = dml.DoubleMLIIVM(
        data, ols(), logit(), logit(), n_folds=N_FOLDS, draw_sample_splitting=False, trimming_threshold=0.05
    )
    iivm.set_sample_splitting(smpls)
    iivm.fit()
    results["IIVM"] = load_style_summary(iivm)
    results["IIVM"]["theta_true"] = 0.5

    # ----- Multi-treatment PLR (manual DGP, shared with Julia) -----
    np.random.seed(SEED + 4)
    n, p = 1000, 6
    X = np.random.randn(n, p)
    b = X[:, 0] + 0.25 * X[:, 1] ** 2
    D = np.column_stack([0.4 * X[:, 0] + np.random.randn(n), 0.4 * X[:, 1] + np.random.randn(n)])
    theta = np.array([0.5, -0.3])
    y = D @ theta + b + np.random.randn(n)
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
    df["y"] = y
    df["d1"] = D[:, 0]
    df["d2"] = D[:, 1]
    df.to_csv(OUT / "plr_multi.csv", index=False)
    smpls = fixed_smpls(n, seed=SEED + 4)
    save_smpls(OUT / "plr_multi_smpls.json", smpls)
    data = dml.DoubleMLData(df, "y", ["d1", "d2"], use_other_treat_as_covariate=True)
    mplr = dml.DoubleMLPLR(data, ols(), ols(), n_folds=N_FOLDS, draw_sample_splitting=False)
    mplr.set_sample_splitting(smpls)
    mplr.fit()
    results["PLR_multi"] = load_style_summary(mplr)
    results["PLR_multi"]["theta_true"] = theta.tolist()

    # ----- Cluster PLR (simple one-way) -----
    np.random.seed(SEED + 5)
    n, n_cl, p = 600, 40, 4
    cluster = np.repeat(np.arange(n_cl), n // n_cl)
    while len(cluster) < n:
        cluster = np.append(cluster, n_cl - 1)
    cluster = cluster[:n]
    X = np.random.randn(n, p)
    a = np.random.randn(n_cl)[cluster]
    d = 0.5 * X[:, 0] + 0.3 * a + np.random.randn(n)
    y = 0.5 * d + X[:, 0] + a + np.random.randn(n)
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
    df["y"] = y
    df["d"] = d
    df["cluster"] = cluster
    df.to_csv(OUT / "plr_cluster.csv", index=False)
    smpls = fixed_smpls(n, seed=SEED + 5)  # note: Python cluster uses cluster folds; we compare iid fit + post SE separately
    save_smpls(OUT / "plr_cluster_smpls.json", smpls)
    data = dml.DoubleMLData(df, "y", "d", cluster_cols="cluster")
    # cluster-aware fit
    cplr = dml.DoubleMLPLR(data, ols(), ols(), n_folds=3)  # cluster folds auto
    cplr.fit()
    results["PLR_cluster"] = load_style_summary(cplr)
    results["PLR_cluster"]["theta_true"] = 0.5
    results["PLR_cluster"]["n_folds_note"] = "Python uses cluster sample splitting (n_folds per cluster=3)"

    # ----- APO contrast framework -----
    np.random.seed(SEED + 1)
    df = pd.read_csv(OUT / "irm.csv")
    dcol = "d"
    ycol = "y"
    xcols = [c for c in df.columns if c not in (ycol, dcol)]
    data = dml.DoubleMLData(df, ycol, dcol, x_cols=xcols)
    smpls = fixed_smpls(len(df), seed=SEED + 1)
    apo0 = dml.DoubleMLAPO(data, ols(), logit(), treatment_level=0.0, n_folds=N_FOLDS, draw_sample_splitting=False, trimming_threshold=0.01)
    apo1 = dml.DoubleMLAPO(data, ols(), logit(), treatment_level=1.0, n_folds=N_FOLDS, draw_sample_splitting=False, trimming_threshold=0.01)
    apo0.set_sample_splitting(smpls)
    apo1.set_sample_splitting(smpls)
    apo0.fit()
    apo1.fit()
    f0 = apo0.construct_framework()
    f1 = apo1.construct_framework()
    ate = f1 - f0
    results["APO_contrast"] = {
        "apo0": f0.thetas.tolist(),
        "apo1": f1.thetas.tolist(),
        "ate": ate.thetas.tolist(),
        "ate_se": ate.ses.tolist(),
        "irm_ate": results["IRM"]["coef"],
    }

    out_path = OUT / "python_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
