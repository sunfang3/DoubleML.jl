# Algorithm Benchmark: Python DoubleML vs Julia DoubleML.jl

| | |
|--|--|
| **Python** | DoubleML `0.11.3` (+ sklearn OLS / LogisticRegression) |
| **Julia** | DoubleML.jl `1.2.0` (LinearRegressionLearner / LogisticRegressionLearner) |
| **Protocol** | Shared CSV; shared K-fold where applicable; `n_rep=1` |
| **Seed** | 3141 |

## 1. Core linear DML (shared folds + OLS) — bit-level

| Model | Py coef | Jl coef | \|Δcoef\| | Py SE | Jl SE |
|-------|--------:|--------:|----------:|------:|------:|
| **PLR** | 0.492395 | 0.492395 | **2e-16** | 0.031448 | 0.031448 |
| **PLIV** | 1.114333 | 1.114333 | **7e-16** | 0.048843 | 0.048843 |
| **PLR multi d1** | 0.488728 | 0.488728 | **2e-16** | 0.031296 | 0.031296 |
| **PLR multi d2** | −0.329201 | −0.329201 | **6e-17** | 0.031463 | 0.031463 |
| **Framework 2×PLR** | 0.984791 | 0.984791 | **3e-16** | 0.062896 | 0.062896 |

**Verdict:** residualization + linear score + SE + Framework algebra match at machine precision.

## 2. Propensity-based models — near agreement

| Model | Py coef | Jl coef | \|Δcoef\| | rel | true | Py bias | Jl bias |
|-------|--------:|--------:|----------:|----:|-----:|--------:|--------:|
| **IRM** | 0.497577 | 0.497578 | **1e-06** | 2e-6 | 0.5 | −0.0024 | −0.0024 |
| **IIVM** | 0.901260 | 0.901265 | **5e-06** | 6e-6 | 0.5 | +0.40 | +0.40 |
| **DID** (two-period) | −0.066439 | −0.066460 | **2e-05** | 3e-4 | — | — | — |

**Verdict:** IRM/DID essentially identical; residual gap is logistic solver. IIVM both far from true LATE on this DGP (weak design) but **agree with each other**.

## 3. PLPR approaches (shared DGP; independent folds)

| Approach | Py coef | Jl coef | \|Δ\| | true | both recover? |
|----------|--------:|--------:|-----:|-----:|:-------------:|
| **fd_exact** | 0.480 | 0.482 | 0.0015 | 0.5 | ✅ |
| **wg_approx** | 0.461 | 0.472 | 0.011 | 0.5 | ✅ |
| **cre_general** | 0.472 | 0.478 | 0.006 | 0.5 | ✅ |
| **cre_normal** | 0.475 | 0.476 | 0.001 | 0.5 | ✅ |

**Verdict:** All four approaches recover θ≈0.5. Small coef gaps expected (no shared sample splits on transformed panel). SE for CRE slightly larger in Julia (fold / SE aggregation differences).

## 4. RDFlex (Python = rdrobust final; Julia = local linear)

| | Py (Conventional) | Julia | \|Δ\| |
|--|------------------:|------:|----:|
| coef | 1.372 | 1.393 | 0.021 |
| SE | 0.416 | 0.403 | 0.013 |

**Verdict:** Close recovery of a positive RD jump; not bit-level (different final estimator / bandwidth).

## 5. DID multi (Callaway–Sant’Anna)

Same CS2021-style panel CSV (Julia: never-treated `d=0`; Python float panel: never-treated `d=+inf`).

| | Python | Julia |
|--|-------:|------:|
| n_ATT | 12 | 12 |
| mean(coef) | ~0.91 | ~0.83 |
| wall time | 0.63 s | 0.13 s |

Per-ATT coefs differ more than linear models (implementation details of gt construction, propensity, and never-treated coding). **Both produce 12 group–time ATTs** on the same design.

## 6. Timing (single wall-clock run)

| Model | Python (s) | Julia (s) | note |
|-------|-----------:|----------:|------|
| PLR | 0.029 | 1.36 | Julia cold JIT on first fit |
| IRM | 0.060 | 2.39 | cold / logistic |
| PLIV | 0.051 | 0.18 | |
| IIVM | 0.155 | 0.60 | |
| PLR multi | 0.080 | 0.002 | warm |
| PLPR (×4 sum) | ~0.18 | ~0.19 | |
| DID | 0.056 | 0.017 | |
| DID multi | 0.625 | 0.131 | |
| RDFlex | 0.45 | 0.61 | |
| **Total** | **~1.7** | **~5.5** | dominated by Julia JIT on first models |

After warmup, Julia OLS models are competitive or faster; first-call compile cost inflates totals.

## 7. Bottom line

| Area | Agreement |
|------|-----------|
| PLR / PLIV / multi / Framework | **Machine precision** |
| IRM / DID / IIVM (cross-impl) | **~1e-6–1e-5** (solver) |
| PLPR fd/wg/cre | **Same recovery of θ**; small numeric gaps |
| RDFlex | **Same order / sign**; ~2% coef gap vs rdrobust |
| DID multi | **Same n_ATT**; moderate per-cell gaps |

**Overall:** main algorithms are numerically aligned for core DML; remaining differences are expected (logistic, RD final stage, multi DID coding).

## Reproduce

```bash
python3 benchmarks/compare_py_jl/run_benchmark_python.py
julia --project=. -e 'using Pkg; Pkg.add(["CSV","JSON"])'
julia --project=. benchmarks/compare_py_jl/run_benchmark_julia.jl
python3 benchmarks/compare_py_jl/compare_benchmark.py
# → benchmarks/compare_py_jl/BENCHMARK_REPORT.md
```
