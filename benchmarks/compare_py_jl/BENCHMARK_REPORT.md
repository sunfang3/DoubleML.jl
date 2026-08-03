# Algorithm Benchmark Round 2 — Python DoubleML vs Julia DoubleML.jl

| | |
|--|--|
| **Python** | DoubleML `0.11.3` |
| **Julia** | DoubleML.jl `1.4.0` |
| **Protocol** | Shared CSV + shared K-fold where possible; OLS / near-unregularized logistic; `n_rep=1` |
| **Seed** | 3141 |
| **Date** | 2026-08-03 |

Coverage: **PLR, IRM, PLIV, IIVM, multi-PLR, Framework, PLPR×4, DID, DID multi, RDFlex, SSM MAR, SSM nonignorable, PLR sensitivity bounds**.

---

## 1. Bit-level agreement (shared folds + OLS)

| Model | Py coef | Jl coef | \|Δcoef\| | Py SE | Jl SE |
|-------|--------:|--------:|----------:|------:|------:|
| **PLR** | 0.492395 | 0.492395 | **~1e-16** | 0.031448 | 0.031448 |
| **PLIV** | 1.114333 | 1.114333 | **~1e-16** | 0.048843 | 0.048843 |
| **PLR multi** d1 | 0.488728 | 0.488728 | **~1e-16** | 0.031296 | 0.031296 |
| **PLR multi** d2 | −0.329201 | −0.329201 | **~1e-16** | 0.031463 | 0.031463 |
| **Framework 2×PLR** | 0.984791 | 0.984791 | **~1e-16** | 0.062896 | 0.062896 |

---

## 2. Sensitivity bounds (same PLR fit, cf_y=0.04, cf_d=0.03)

| quantity | Python | Julia | \|Δ\| |
|----------|-------:|------:|----:|
| theta_lower | 0.457622 | 0.457622 | **~1e-16** |
| theta_upper | 0.527168 | 0.527168 | **~1e-16** |
| ci_lower | 0.405879 | 0.405879 | **~1e-16** |
| ci_upper | 0.578944 | 0.578944 | **~1e-16** |
| **rv** | 0.389238 | 0.389238 | **~3e-7** |
| **rva** | 0.354341 | 0.354340 | **~5e-7** |

**Verdict:** OVB sensitivity algebra matches Python at numerical precision (RV/RVa differ only at golden-section search tolerance).

---

## 3. Propensity / classification models

| Model | Py coef | Jl coef | \|Δcoef\| | true | notes |
|-------|--------:|--------:|----------:|-----:|-------|
| **IRM** | 0.497577 | 0.497578 | **1e-6** | 0.5 | excellent |
| **DID** | −0.066439 | −0.066460 | **2e-5** | — | excellent |
| **IIVM** | 0.901260 | 0.901265 | **5e-6** | 0.5 | agree; both far from true LATE (weak DGP) |
| **SSM MAR** | 0.883 | 0.891 | 0.008 | 1.0 | good (~1% rel) |
| **SSM nonignorable** | 1.073 | 0.970 | 0.10 | 1.0 | larger gap (nested half-split RNG) |

---

## 4. PLPR (four approaches, shared DGP)

| Approach | Py | Jl | \|Δ\| | recovery of θ=0.5 |
|----------|---:|---:|----:|:-----------------:|
| fd_exact | 0.480 | 0.482 | 0.0015 | ✅ |
| wg_approx | 0.461 | 0.472 | 0.011 | ✅ |
| cre_general | 0.472 | 0.478 | 0.006 | ✅ |
| cre_normal | 0.475 | 0.476 | 0.001 | ✅ |

---

## 5. RDFlex & DID multi

| Model | Python | Julia | \|Δcoef\| |
|-------|-------:|------:|----------:|
| **RDFlex** (sharp) | 1.372 | 1.393 | 0.021 (~1.5%) |
| **DID multi** n_ATT | 12 | 12 | — |
| DID multi mean(coef) | ~0.91 | ~0.83 | moderate per-cell gaps |

RDFlex: Python final stage = `rdrobust`; Julia = local linear + residual ROT.  
DID multi: same panel design; never-treated coding (0 vs +inf) and CS internals differ.

---

## 6. Timing (single wall-clock run)

| | Python | Julia |
|--|-------:|------:|
| **Total** | ~2.4 s | ~9.0 s |

Julia totals dominated by **JIT cold start** on first large fits (PLR/IRM). Warm OLS multi/PLPR are often faster than Python.

---

## 7. Bottom line (Round 2)

| Area | Agreement |
|------|-----------|
| Core linear DML + Framework | **Machine precision** |
| Sensitivity bounds / RV | **Machine precision / ~1e-7** |
| IRM / DID / IIVM (cross-impl) | **~1e-6–1e-5** |
| SSM MAR | **~1%** coef gap |
| SSM nonignorable | **~10%** (nested split RNG; both near truth 1.0) |
| PLPR ×4 | **Recover θ; small fold gaps** |
| RDFlex | **~2%** vs rdrobust |
| DID multi | **Same n_ATT; moderate cell gaps** |

**Overall:** v1.4.0 remains tightly aligned with Python on core estimators and sensitivity; residual gaps are expected (logistic, nested CF RNG, rdrobust vs local linear, DID multi coding).

---

## Reproduce

```bash
python3 benchmarks/compare_py_jl/run_benchmark_python.py
julia --project=. -e 'using Pkg; Pkg.add(["CSV","JSON"])'
julia --project=. benchmarks/compare_py_jl/run_benchmark_julia.jl
python3 benchmarks/compare_py_jl/compare_benchmark.py
```
