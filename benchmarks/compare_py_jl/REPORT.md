# Numerical comparison: Python DoubleML 0.11.3 vs Julia DoubleML.jl 1.1.0

**Date:** 2026-08-03  
**Protocol:** same CSV data, same K-fold indices (Python 0-based → Julia 1-based),  
learners = OLS (`LinearRegression` / `LinearRegressionLearner`), logistic ≈ unregularized,  
`n_rep=1`, shared `set_sample_splitting`.

Reproduce:

```bash
# Python
python3 benchmarks/compare_py_jl/run_python.py
# Julia (needs CSV+JSON in env, or run with temp deps)
julia --project=. -e 'using Pkg; Pkg.add(["CSV","JSON"])'
julia --project=. benchmarks/compare_py_jl/run_julia.jl
```

## Main results (coef / SE)

| Model | j | Python coef | Julia coef | \|Δcoef\| | Python SE | Julia SE | \|ΔSE\| |
|-------|---|------------:|-----------:|----------:|----------:|---------:|--------:|
| **PLR** | 0 | 0.46523734 | 0.46523734 | **1e-16** | 0.036183 | 0.036183 | **0** |
| **IRM** | 0 | 0.58605081 | 0.58614205 | **9e-5** | 0.132910 | 0.132973 | **6e-5** |
| **PLIV** | 0 | 1.04832816 | 1.04832816 | **4e-16** | 0.064907 | 0.064907 | **0** |
| **IIVM** | 0 | −0.41440226 | −0.41439575 | **7e-6** | 0.844356 | 0.844410 | **5e-5** |
| **PLR multi d1** | 0 | 0.44375434 | 0.44375434 | **0** | 0.030484 | 0.030484 | **0** |
| **PLR multi d2** | 1 | −0.33219297 | −0.33219297 | **0** | 0.030550 | 0.030550 | **0** |
| **PLR cluster** | 0 | 0.78685529 | 0.78582430 | 1e-3 | 0.057469 | 0.058672 | 1e-3 |
| **Framework PLR** | 0 | 0.46523734 | 0.46523734 | **1e-16** | 0.036183 | 0.036183 | **0** |
| **Framework 2×PLR** | 0 | 0.93047467 | 0.93047467 | **2e-16** | 0.072367 | 0.072367 | **0** |

## Framework / APO checks

| Check | Python | Julia |
|-------|--------|-------|
| APO(1)−APO(0) vs IRM ATE | match to ~1e-16 | match to ~1e-16 |
| Framework 2× scales SE by 2 | yes | yes |

## Interpretation

### Machine-precision agreement (linear residual-on-residual)
- **PLR**, **PLIV**, **multi-treatment PLR**, **Framework** (+ scale): differences at floating-point noise.  
  → Cross-fit residualization + linear score solve + SE formula are aligned when folds and OLS match.

### Near agreement (propensity / logistic)
- **IRM**, **IIVM**, **APO contrast**: relative coef gap ~1e-4–1e-5.  
  → Driven by logistic solver differences (sklearn `LogisticRegression` vs Julia IRLS), not DML score algebra.  
  APO contrast equals IRM ATE on each side (internal consistency of Framework `f1 - f0`).

### Cluster-in-fit (expected gap)
- **PLR cluster**: ~0.1% coef gap, ~2% SE gap.  
  → Cluster **fold partitions were not synchronized** (Python auto cluster KFold RNG ≠ Julia `MersenneTwister`).  
  Both recover large SE vs iid and `var_scaling = n_clusters = 40` on Julia.  
  Full numerical identity needs exporting Python `smpls` + `smpls_cluster`.

### DGP recovery (both packages)
- PLR/PLIV close to truth; IRM noisier at n=1000; IIVM unstable on this seed (both packages, huge SE) — weak LATE design, not Julia-specific.

## Verdict

| Area | Alignment |
|------|-----------|
| Linear DML core (PLR/PLIV/multi) | **Excellent** (bit-level with shared folds + OLS) |
| Framework algebra | **Excellent** |
| Binary propensity models (IRM/IIVM/APO) | **Very good** (solver-limited) |
| Cluster-in-fit | **Structurally OK**; fold-sync needed for bit match |
| Overall P0 claims | **Supported by numerical evidence** |

## Artifacts
- `data/python_results.json`, `data/julia_results.json`
- Shared CSVs + `*_smpls.json`
- Scripts: `run_python.py`, `run_julia.jl`
