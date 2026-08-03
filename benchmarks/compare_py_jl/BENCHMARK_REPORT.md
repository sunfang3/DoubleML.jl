# Algorithm Benchmark: Python DoubleML vs Julia DoubleML.jl

- **Python**: DoubleML `0.11.3`
- **Julia**: DoubleML.jl `1.4.0`
- **Protocol**: shared CSV + shared K-fold (where applicable); OLS / near-unregularized logistic; `n_rep=1`
- **Timing protocol**: `warm_median` (discard 1 cold fit; median of 3 warm fits). `seconds` = warm median.
- **Seed**: 3141, default `n_folds=5`
- **Runtime gate**: `(t_jl − t_py) / t_py ≤ 0.30` on **sum of warm** seconds over timed main models (bookkeeping rows excluded).

## Coefficient & SE comparison

| Model | j | Py coef | Jl coef | |Δcoef| | Py SE | Jl SE | |ΔSE| | rel Δcoef | Py warm s | Jl warm s | speedup |
|-------|---|--------:|--------:|--------:|------:|------:|------:|----------:|----------:|----------:|--------:|
| PLR | 0 | 0.492395 | 0.492395 | 1.7e-16 | 0.0314478 | 0.0314478 | 1.4e-17 | 3.4e-16 | 0.0265 | 0.000826 | 32 |
| IRM | 0 | 0.497577 | 0.497578 | 1.1e-06 | 0.104578 | 0.104559 | 1.9e-05 | 2.3e-06 | 0.0741 | 0.00211 | 35 |
| PLIV | 0 | 1.11433 | 1.11433 | 6.7e-16 | 0.0488425 | 0.0488425 | 1e-16 | 6e-16 | 0.0459 | 0.00136 | 34 |
| IIVM | 0 | 0.90126 | 0.901265 | 5e-06 | 0.394506 | 0.394486 | 1.9e-05 | 5.6e-06 | 0.192 | 0.00474 | 41 |
| DID_multi | 0 | -0.0212575 | 0.146854 | 0.17 | 0.372017 | 0.333789 | 0.038 | 7.9 | 0.613 | 0.00379 | 1.6e+02 |
|  | 1 | 1.1167 | 0.841036 | 0.28 | 0.31948 | 0.444913 | 0.13 | 0.25 | — | — | — |
|  | 2 | 1.34926 | 1.16999 | 0.18 | 0.419557 | 0.60978 | 0.19 | 0.13 | — | — | — |
|  | 3 | 2.55878 | 2.5724 | 0.014 | 0.480359 | 0.431566 | 0.049 | 0.0053 | — | — | — |
|  | 4 | -0.0789801 | -0.268773 | 0.19 | 0.376616 | 0.424971 | 0.048 | 2.4 | — | — | — |
|  | 5 | 0.136747 | -0.0353716 | 0.17 | 0.381567 | 0.38557 | 0.004 | 1.3 | — | — | — |
|  | 6 | 1.15217 | 1.36248 | 0.21 | 0.43779 | 0.402696 | 0.035 | 0.18 | — | — | — |
|  | 7 | 2.10181 | 2.47355 | 0.37 | 0.43152 | 0.446293 | 0.015 | 0.18 | — | — | — |
|  | 8 | 0.0507919 | -0.110592 | 0.16 | 0.333848 | 0.327067 | 0.0068 | 3.2 | — | — | — |
|  | 9 | 1.06738 | -0.320231 | 1.4 | 0.834284 | 0.362968 | 0.47 | 1.3 | — | — | — |
|  | 10 | -0.260874 | 0.0886433 | 0.35 | 0.353096 | 0.333273 | 0.02 | 1.3 | — | — | — |
|  | 11 | 1.42751 | 1.08204 | 0.35 | 0.419571 | 0.344585 | 0.075 | 0.24 | — | — | — |
| PLR_multi | 0 | 0.488728 | 0.488728 | 1.7e-16 | 0.0312964 | 0.0312964 | 6.9e-18 | 3.4e-16 | 0.0555 | 0.00174 | 32 |
|  | 1 | -0.329201 | -0.329201 | 5.6e-17 | 0.0314631 | 0.0314631 | 1.4e-17 | 1.7e-16 | — | — | — |
| PLPR_cre_general | 0 | 0.465036 | 0.477954 | 0.013 | 0.0312368 | 0.0536894 | 0.022 | 0.028 | 0.0414 | 0.000925 | 45 |
| PLPR_cre_normal | 0 | 0.472287 | 0.47601 | 0.0037 | 0.0317692 | 0.054198 | 0.022 | 0.0079 | 0.0388 | 0.00092 | 42 |
| PLPR_fd_exact | 0 | 0.481885 | 0.481669 | 0.00022 | 0.0353651 | 0.0345256 | 0.00084 | 0.00045 | 0.0498 | 0.00448 | 11 |
| PLPR_wg_approx | 0 | 0.47427 | 0.47169 | 0.0026 | 0.0318003 | 0.0319384 | 0.00014 | 0.0054 | 0.0409 | 0.000543 | 75 |
| DID | 0 | -0.0664388 | -0.0664597 | 2.1e-05 | 0.106252 | 0.106256 | 4.5e-06 | 0.00031 | 0.0882 | 0.00193 | 46 |
| RDFlex | 0 | 1.31615 | 1.39342 | 0.077 | 0.409226 | 0.403125 | 0.0061 | 0.059 | 0.0494 | 0.00102 | 48 |
| Framework_2x_PLR | 0 | 0.98479 | 0.98479 | 3.3e-16 | 0.0628956 | 0.0628956 | 2.8e-17 | 3.4e-16 | 0 | 0 | — |
| PLR_sensitivity | 0 | 0.492395 | 0.492395 | 1.7e-16 | 0.0314478 | 0.0314478 | 1.4e-17 | 3.4e-16 | 0 | 0 | — |
| SSM_MAR | 0 | 0.882975 | 0.891248 | 0.0083 | 0.08832 | 0.0860744 | 0.0022 | 0.0094 | 0.161 | 0.00505 | 32 |
| SSM_nonignorable | 0 | 1.07283 | 0.969675 | 0.1 | 0.172257 | 0.086096 | 0.086 | 0.096 | 0.288 | 0.00376 | 77 |

## Bias vs true parameter (where known)

| Model | j | true | Py bias | Jl bias |
|-------|---|-----:|--------:|--------:|
| PLR | 0 | 0.5 | -0.00760475 | -0.00760475 |
| IRM | 0 | 0.5 | -0.00242336 | -0.00242224 |
| PLIV | 0 | 1 | 0.114333 | 0.114333 |
| IIVM | 0 | 0.5 | 0.40126 | 0.401265 |
| PLR_multi | 0 | 0.5 | -0.0112715 | -0.0112715 |
|  | 1 | -0.3 | -0.0292012 | -0.0292012 |
| PLPR_cre_general | 0 | 0.5 | -0.0349639 | -0.0220458 |
| PLPR_cre_normal | 0 | 0.5 | -0.0277131 | -0.0239896 |
| PLPR_fd_exact | 0 | 0.5 | -0.0181155 | -0.0183313 |
| PLPR_wg_approx | 0 | 0.5 | -0.0257304 | -0.0283103 |
| Framework_2x_PLR | 0 | 1 | -0.0152095 | -0.0152095 |
| SSM_MAR | 0 | 1 | -0.117025 | -0.108752 |
| SSM_nonignorable | 0 | 1 | 0.0728304 | -0.0303247 |

## Summary (accuracy)

- Median relative |Δcoef| (excl. DID multi): **3.14e-04**
- Max relative |Δcoef| (excl. DID multi): **9.62e-02**
- Median |Δcoef| (excl. DID multi): **2.08e-05**
- Main-model cells within 30% rel Δ: **16/16**
- DID multi: n_ATT=12, median cell rel=7.53e-01, frac ≤30%=50%

## Runtime (warm multi-rep gate)

| Aggregate | Python | Julia |
|-----------|-------:|------:|
| **Warm total** (gating) | **1.7653s** | **0.0332s** |
| Cold total (discarded) | 2.1179s | 8.6078s |
| Ratio t_jl / t_py (warm) | — | **0.019×** |
| Relative gap (t_jl − t_py)/t_py | — | **-98.12%** |
| Gate (≤ 30%) | — | **PASS** |

**RUNTIME_GAP_PASS: true**

Per-model warm seconds:

| Model | Py warm | Jl warm | Py cold | Jl cold | Jl/Py |
|-------|--------:|--------:|--------:|--------:|------:|
| DID | 0.08817 | 0.001931 | 0.08909 | 0.03118 | 0.022 |
| DID_multi | 0.6134 | 0.003788 | 0.7056 | 0.1345 | 0.0062 |
| IIVM | 0.1919 | 0.004738 | 0.15 | 0.626 | 0.025 |
| IRM | 0.07406 | 0.002112 | 0.05862 | 2.543 | 0.029 |
| PLIV | 0.04585 | 0.00136 | 0.05162 | 0.3539 | 0.03 |
| PLPR_cre_general | 0.04137 | 0.0009251 | 0.04297 | 0.00091 | 0.022 |
| PLPR_cre_normal | 0.03882 | 0.0009201 | 0.03718 | 0.000798 | 0.024 |
| PLPR_fd_exact | 0.04975 | 0.00448 | 0.04658 | 0.1816 | 0.09 |
| PLPR_wg_approx | 0.04086 | 0.0005429 | 0.04028 | 0.0007129 | 0.013 |
| PLR | 0.02651 | 0.0008259 | 0.03255 | 1.368 | 0.031 |
| PLR_multi | 0.05548 | 0.001737 | 0.07841 | 0.002602 | 0.031 |
| RDFlex | 0.04936 | 0.001021 | 0.4333 | 0.6428 | 0.021 |
| SSM_MAR | 0.1612 | 0.005049 | 0.1337 | 0.4691 | 0.031 |
| SSM_nonignorable | 0.2885 | 0.003762 | 0.218 | 2.252 | 0.013 |

## Sensitivity bounds (PLR, cf_y=0.04, cf_d=0.03)

| quantity | Python | Julia | |Δ| |
|----------|-------:|------:|----:|
| theta_lower | 0.457622 | 0.457622 | 1.67e-16 |
| theta_upper | 0.527168 | 0.527168 | 2.22e-16 |
| ci_lower | 0.405879 | 0.405879 | 1.67e-16 |
| ci_upper | 0.578944 | 0.578944 | 2.22e-16 |
| rv | 0.389238 | 0.389238 | 2.56e-07 |
| rva | 0.354341 | 0.35434 | 5.49e-07 |

## Notes

- **Linear models (PLR/PLIV/multi/PLPR/Framework)**: expect near bit-level agreement with shared folds + OLS.
- **IRM/IIVM/DID/SSM**: small gaps from logistic solver / nested CF RNG differences.
- **SSM nonignorable**: nested half-splits use stratified shuffle; Julia RNG ≠ sklearn `random_state=42` unless aligned — expect larger gaps than MAR.
- **RDFlex**: Python uses `rdrobust` final stage; Julia uses weighted local linear + residual ROT — coef may differ more than linear DML.
- **DID multi**: never-treated coding (0 vs +inf) and CS internals may differ per-cell ATTs.
- **Timing**: `seconds` is the **median of warm fits** after discarding cold/JIT. Cold is reported separately and is **not** the 30% gate.
- Framework / sensitivity rows are bookkeeping (re-use fit; excluded from runtime total).

## Reproduce

```bash
python3 benchmarks/compare_py_jl/run_benchmark_python.py
julia --project=. benchmarks/compare_py_jl/run_benchmark_julia.jl
python3 benchmarks/compare_py_jl/compare_benchmark.py
```
