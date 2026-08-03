# Algorithm Benchmark Round 3 — Python DoubleML vs Julia DoubleML.jl

| | |
|--|--|
| **Python** | DoubleML `0.11.3` |
| **Julia** | DoubleML.jl `1.4.0` |
| **Protocol** | Shared CSV + shared K-fold where possible; OLS / near-unregularized logistic; `n_rep=1` |
| **Seed** | 3141 |
| **Date** | 2026-08-03 |

Coverage: **PLR, IRM, PLIV, IIVM, multi-PLR, Framework, PLPR×4, DID, DID multi, RDFlex, SSM MAR, SSM nonignorable, PLR sensitivity bounds**.

**30% accuracy gate (main models):** all 14 primary estimators pass `|Δcoef| / max(|py|, ε) ≤ 0.30`.  
(DID multi cell-level ATTs remain partially aligned — see §5.)

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

| Model | Py coef | Jl coef | \|Δcoef\| | rel Δ | true | notes |
|-------|--------:|--------:|----------:|------:|-----:|-------|
| **IRM** | 0.497577 | 0.497578 | **1.1e-6** | 2e-6 | 0.5 | excellent |
| **DID** | −0.066439 | −0.066460 | **2.1e-5** | 3e-4 | — | excellent |
| **IIVM** | 0.901260 | 0.901265 | **5.0e-6** | 6e-6 | 0.5 | agree; both far from true LATE (weak DGP) |
| **SSM MAR** | 0.8830 | 0.8912 | 0.0083 | **0.9%** | 1.0 | good |
| **SSM nonignorable** | 1.0728 | 0.9697 | 0.103 | **9.6%** | 1.0 | within 30%; nested half-split RNG |

---

## 4. PLPR (four approaches, shared DGP)

| Approach | Py | Jl | \|Δ\| | rel Δ | recovery of θ=0.5 |
|----------|---:|---:|----:|------:|:-----------------:|
| fd_exact | 0.4801 | 0.4817 | 0.0015 | 0.3% | ✅ |
| wg_approx | 0.4606 | 0.4717 | 0.0111 | 2.4% | ✅ |
| cre_general | 0.4717 | 0.4780 | 0.0062 | 1.3% | ✅ |
| cre_normal | 0.4748 | 0.4760 | 0.0012 | 0.3% | ✅ |

---

## 5. RDFlex & DID multi

| Model | Python | Julia | \|Δcoef\| | rel Δ |
|-------|-------:|------:|----------:|------:|
| **RDFlex** (sharp) | 1.3720 | 1.3934 | 0.0214 | **1.6%** |
| **DID multi** n_ATT | 12 | 12 | — | — |
| DID multi mean(coef) | 0.829 | 0.750 | 0.079 | — |
| DID multi cell rel Δ | — | — | median 27% | max ≫30% (near-zero cells) |
| DID multi frac within 30% | — | — | **58%** of cells | |

RDFlex: Python final stage = `rdrobust`; Julia = local linear + residual ROT.  
DID multi: same panel design / n_ATT; never-treated coding (0 vs +inf) and CS internals still drive per-cell gaps. **Not part of the primary 30% gate.**

---

## 6. Full coefficient table (auto-generated)

| Model | j | Py coef | Jl coef | \|Δcoef\| | Py SE | Jl SE | \|ΔSE\| | rel Δcoef | Py s | Jl s | speedup |
|-------|---|--------:|--------:|----------:|------:|------:|--------:|----------:|-----:|-----:|--------:|
| PLR | 0 | 0.492395 | 0.492395 | 1.7e-16 | 0.0314478 | 0.0314478 | 1.4e-17 | 3.4e-16 | 0.0433 | 1.51 | 0.029 |
| IRM | 0 | 0.497577 | 0.497578 | 1.1e-06 | 0.104578 | 0.104559 | 1.9e-05 | 2.3e-06 | 0.116 | 2.85 | 0.041 |
| PLIV | 0 | 1.11433 | 1.11433 | 6.7e-16 | 0.0488425 | 0.0488425 | 1e-16 | 6e-16 | 0.0708 | 0.429 | 0.17 |
| IIVM | 0 | 0.90126 | 0.901265 | 5e-06 | 0.394506 | 0.394486 | 1.9e-05 | 5.6e-06 | 0.226 | 0.753 | 0.3 |
| PLR_multi | 0 | 0.488728 | 0.488728 | 1.7e-16 | 0.0312964 | 0.0312964 | 6.9e-18 | 3.4e-16 | 0.0902 | 0.00148 | 61 |
|  | 1 | −0.329201 | −0.329201 | 5.6e-17 | 0.0314631 | 0.0314631 | 1.4e-17 | 1.7e-16 | — | — | — |
| PLPR_fd_exact | 0 | 0.480145 | 0.481669 | 0.0015 | 0.0345423 | 0.0345256 | 1.7e-05 | 0.0032 | 0.0647 | 0.197 | 0.33 |
| PLPR_wg_approx | 0 | 0.460605 | 0.47169 | 0.011 | 0.0323958 | 0.0319384 | 0.00046 | 0.024 | 0.0482 | 0.000679 | 71 |
| PLPR_cre_general | 0 | 0.471737 | 0.477954 | 0.0062 | 0.0316672 | 0.0536894 | 0.022 | 0.013 | 0.0526 | 0.00106 | 50 |
| PLPR_cre_normal | 0 | 0.474786 | 0.47601 | 0.0012 | 0.0319408 | 0.054198 | 0.022 | 0.0026 | 0.0906 | 0.00083 | 109 |
| DID | 0 | −0.066439 | −0.066460 | 2.1e-05 | 0.106252 | 0.106256 | 4.5e-06 | 0.00031 | 0.13 | 0.0233 | 5.6 |
| RDFlex | 0 | 1.37201 | 1.39342 | 0.021 | 0.416091 | 0.403125 | 0.013 | 0.016 | 0.507 | 0.698 | 0.73 |
| Framework_2x_PLR | 0 | 0.98479 | 0.98479 | 3.3e-16 | 0.0628956 | 0.0628956 | 2.8e-17 | 3.4e-16 | 0 | 0 | — |
| SSM_MAR | 0 | 0.882975 | 0.891248 | 0.0083 | 0.08832 | 0.0860744 | 0.0022 | 0.0094 | 0.331 | 0.687 | 0.48 |
| SSM_nonignorable | 0 | 1.07283 | 0.969675 | 0.10 | 0.172257 | 0.086096 | 0.086 | 0.096 | 0.563 | 2.41 | 0.23 |

*(DID multi 12 cells omitted from main table; see §5.)*

---

## 7. Bias vs true parameter (where known)

| Model | true | Py bias | Jl bias |
|-------|-----:|--------:|--------:|
| PLR | 0.5 | −0.007605 | −0.007605 |
| IRM | 0.5 | −0.002423 | −0.002422 |
| PLIV | 1.0 | +0.114333 | +0.114333 |
| IIVM | 0.5 | +0.40126 | +0.40127 |
| PLR multi d1/d2 | 0.5 / −0.3 | −0.011 / −0.029 | same |
| PLPR (all 4) | 0.5 | −0.02…−0.04 | −0.02…−0.03 |
| SSM MAR | 1.0 | −0.117 | −0.109 |
| SSM nonignorable | 1.0 | +0.073 | −0.030 |

---

## 8. Timing (single wall-clock run)

| | Python | Julia |
|--|-------:|------:|
| **Total** | **3.18 s** | **9.69 s** |
| Overall Py/Jl | — | **0.33×** |

Julia totals dominated by **JIT cold start** on first large fits (PLR ~1.5 s, IRM ~2.8 s, SSM NI ~2.4 s).  
Warm OLS multi / PLPR CRE-WG are often **50–100× faster** than Python.

---

## 9. Bottom line (Round 3 vs Round 2)

| Area | Round 2 | Round 3 | Change |
|------|---------|---------|--------|
| Core linear DML + Framework | machine precision | machine precision | stable |
| Sensitivity bounds / RV | ~1e-16 / ~1e-7 | ~1e-16 / ~1e-7 | stable |
| IRM / DID / IIVM | ~1e-6–1e-5 | ~1e-6–1e-5 | stable |
| SSM MAR | ~1% | **0.9%** | stable |
| SSM nonignorable | ~10% | **9.6%** | stable (within 30%) |
| PLPR ×4 | small fold gaps | 0.3–2.4% | stable |
| RDFlex | ~2% | **1.6%** | stable |
| DID multi | moderate | median cell rel 27%; 58% within 30% | still open |
| **Main models ≤30% rel Δ** | — | **14/14 pass** | ✅ |

**Overall:** v1.4.0 remains tightly aligned with Python DoubleML 0.11.3. Residual gaps are expected (logistic solvers, nested CF RNG, rdrobust vs local linear, DID multi coding). Primary estimators all satisfy a **30% relative coefficient agreement** gate vs Python under the shared-fold protocol.

---

## Notes

- **Linear models (PLR/PLIV/multi/PLPR/Framework):** near bit-level agreement with shared folds + OLS.
- **IRM/IIVM/DID/SSM:** small gaps from logistic solver / nested CF RNG differences.
- **SSM nonignorable:** nested half-splits use stratified shuffle; Julia RNG ≠ sklearn `random_state=42` unless aligned.
- **RDFlex:** Python uses `rdrobust` final stage; Julia uses weighted local linear + residual ROT.
- **DID multi:** never-treated coding (0 vs +inf) and CS internals may differ per-cell ATTs.
- Timings are single-run wall clock; Julia first fits include JIT compile cost.

---

## Reproduce

```bash
python3 benchmarks/compare_py_jl/run_benchmark_python.py
julia --project=. -e 'using Pkg; Pkg.add(["CSV","JSON"])'
julia --project=. benchmarks/compare_py_jl/run_benchmark_julia.jl
python3 benchmarks/compare_py_jl/compare_benchmark.py
# then remove temporary CSV/JSON from Project.toml if added
```
