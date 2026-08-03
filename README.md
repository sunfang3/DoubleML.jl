# DoubleML.jl

Julia implementation of **Double / Debiased Machine Learning**
([Chernozhukov et al., 2018](https://doi.org/10.1111/ectj.12097)), designed after the
Python package [DoubleML](https://docs.doubleml.org/)
([doubleml-for-py](https://github.com/DoubleML/doubleml-for-py)).

> Target: **Julia ≥ 1.10** (developed & tested on **Julia 1.12**).

## Design goals

| Goal | Choice |
|------|--------|
| API familiarity | Mirror Python `DoubleMLData` / PLR / IRM / **PLIV** / **IIVM** / `fit` / `summary_table` / `confint` |
| Long-term ML stack | Duck-typed learners + **DecisionTree.jl** random forests (pure Julia, widely used) |
| Lightweight core | Closed-form **ridge** & **logistic** (no heavyweight MLJ install required to run) |
| Extensibility | Any object with `fit!` / `predict` / `clone` (and `predict_proba` for classifiers) |

MLJ.jl can be wrapped behind the same learner protocol later; we deliberately keep it
optional so the package installs and tests quickly while staying compatible with the
JuliaAI ecosystem.

## Install (dev)

```julia
using Pkg
Pkg.develop(path="path/to/DoubleML")   # or Pkg.add(url="https://github.com/sunfang3/DoubleML.jl")
```

## Quick start

```julia
using DoubleML

# --- PLR (partially linear regression) ---
data = make_plr_data(n_obs=500, dim_x=20, theta=0.5; seed=3141)
dml = DoubleMLPLR(data, RidgeLearner(α=1.0), RidgeLearner(α=1.0); n_folds=5)
fit!(dml)
summary_table(dml)
confint(dml)

# --- IRM (binary treatment ATE) ---
data_irm = make_irm_data(n_obs=800, dim_x=10, theta=0.5; seed=42)
irm = DoubleMLIRM(
    data_irm,
    RandomForestRegressorLearner(n_trees=100, max_depth=6),
    LogisticRegressionLearner(α=1.0);
    n_folds=5,
)
fit!(irm)
summary_table(irm)

# --- PLIV (partially linear IV) ---
data_pliv = make_pliv_data(n_obs=800, dim_x=15, dim_z=1, theta=0.5; seed=99)
ml = RidgeLearner(α=0.5)
pliv = DoubleMLPLIV(data_pliv, clone(ml), clone(ml), clone(ml); n_folds=5)
fit!(pliv)
summary_table(pliv)

# --- IIVM (LATE: binary D, binary Z) ---
data_iivm = make_iivm_data(n_obs=1500, dim_x=8, theta=0.5; seed=7)
iivm = DoubleMLIIVM(
    data_iivm,
    RidgeLearner(α=0.5),
    LogisticRegressionLearner(α=0.5),
    LogisticRegressionLearner(α=0.5);
    n_folds=5,
)
fit!(iivm)
summary_table(iivm)
```

## Architecture (Python → Julia)

```
Python DoubleML                     Julia DoubleML.jl
─────────────────                   ─────────────────
doubleml.DoubleMLData          →    DoubleMLData
doubleml.DoubleML  (ABC)       →    AbstractDoubleML
doubleml.DoubleMLPLR           →    DoubleMLPLR
doubleml.DoubleMLIRM           →    DoubleMLIRM
doubleml.DoubleMLPLIV          →    DoubleMLPLIV
doubleml.DoubleMLIIVM          →    DoubleMLIIVM
sklearn estimator              →    fit! / predict / predict_proba / clone
model.fit()                    →    fit!(model)
model.summary                  →    summary_table(model)
model.confint()                →    confint(model)
```

### Implemented scores

| Model | Score | Status |
|-------|-------|--------|
| PLR | `partialling out` | ✅ |
| PLR | `IV-type` | ✅ |
| IRM | `ATE` (doubly robust) | ✅ |
| IRM | `ATTE` | ✅ (experimental) |
| **PLIV** | `partialling out` (`:partialX`, 1 or multi Z) | ✅ |
| **PLIV** | `IV-type` (single Z, `:partialX`) | ✅ |
| **PLIV** | `:partialZ` / `:partialXZ` | ✅ |
| **IIVM** | `LATE` | ✅ |
| **Multiplier bootstrap** | `normal` / `Bayes` / `wild` + joint CI | ✅ |
| **Hyperparameter tuning** | grid / random search via `tune!` | ✅ |
| **Sensitivity analysis** | OVB bounds (cf_y, cf_d, rho), RV / RVa (PLR, IRM) | ✅ |

## Built-in learners

| Learner | Role | Backend |
|---------|------|---------|
| `RidgeLearner` / `LinearRegressionLearner` | regression | closed form |
| `LogisticRegressionLearner` | binary propensity | IRLS + L2 |
| `RandomForestRegressorLearner` | flexible regression | DecisionTree.jl |
| `RandomForestClassifierLearner` | flexible propensity | DecisionTree.jl |

### Custom learners

```julia
mutable struct MyLearner <: DoubleML.AbstractLearner
    # fields...
end
DoubleML.clone(m::MyLearner) = MyLearner(...)
function DoubleML.fit!(m::MyLearner, X, y) ...; return m; end
function DoubleML.predict(m::MyLearner, X) ...; end
# for classifiers:
DoubleML.is_classifier(::MyLearner) = true
function DoubleML.predict_proba(m::MyLearner, X) ...; end
```

## Project layout

```
DoubleML/
├── Project.toml
├── src/
│   ├── DoubleML.jl      # module entry
│   ├── learners.jl      # ridge / logistic / RF
│   ├── data.jl          # DoubleMLData
│   ├── sample_splitting.jl
│   ├── base.jl          # coef aggregation, SE, confint, summary
│   ├── plr.jl
│   ├── irm.jl
│   ├── pliv.jl          # partially linear IV
│   ├── iivm.jl          # interactive IV (LATE)
│   ├── tune.jl          # grid / random hyperparameter search
│   ├── sensitivity.jl   # OVB sensitivity (Chernozhukov et al. 2022)
│   └── datasets.jl
├── test/runtests.jl
├── examples/plr_irm_demo.jl
└── README.md
```

## IV models in brief

**PLIV** — continuous treatment, instrument(s) `Z` in `DoubleMLData`:

```julia
data = make_pliv_data(n_obs=1000, dim_z=1, theta=0.5; seed=1)
ml = RidgeLearner(α=0.5)
DoubleMLPLIV(data, clone(ml), clone(ml), clone(ml))          # :partialX (default)
DoubleMLPLIV_partialZ(data, clone(ml))                        # :partialZ
DoubleMLPLIV_partialXZ(data, clone(ml), clone(ml), clone(ml)) # :partialXZ
```

**IIVM** — binary `D` and binary `Z`, target = LATE:

```julia
data = make_iivm_data(n_obs=2000, theta=0.5; seed=1)
DoubleMLIIVM(data, ml_g, ml_m, ml_r)  # E[Y|X,Z], E[Z|X], E[D|X,Z]
```

**Multiplier bootstrap** (joint CIs, Python-compatible):

```julia
fit!(dml)
bootstrap!(dml; method="normal", n_rep_boot=500)  # also "Bayes", "wild"
confint(dml; joint=true)
```

**Hyperparameter tuning** (Python `tune` analogue):

```julia
dml = DoubleMLPLR(data, RidgeLearner(α=1.0), RidgeLearner(α=1.0); n_folds=5)
tune!(dml; param_grids=Dict(
    :ml_l => Dict(:α => [0.01, 0.1, 1.0, 10.0]),
    :ml_m => Dict(:α => [0.01, 0.1, 1.0, 10.0]),
), n_folds_tune=5, search_mode=:grid)   # or :random with n_iter=...
fit!(dml)
summary_table(dml)

# Also works for IRM / PLIV / IIVM with the corresponding learner keys.
# Low-level: tune a single learner
best, res = tune_learner(RidgeLearner(), X, y, Dict(:α => [0.1, 1.0, 10.0]))
```

**Sensitivity analysis** (omitted variable bias; Chernozhukov et al. 2022; PLR & IRM):

```julia
fit!(dml)
sensitivity_analysis!(dml; cf_y=0.04, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0)
println(sensitivity_summary(dml))
# dml.sensitivity.theta_lower / theta_upper / rv / rva

# Benchmark confounders by comparing long vs short regressions:
# bm = sensitivity_benchmark(dml_long, dml_short)  # → (cf_y, cf_d, rho, delta_theta)
```

## Run tests

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## References

- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*.
- Bach, P. et al. (2022). DoubleML — An object-oriented implementation of double machine learning in Python. *JMLR*.
- Official Python/R packages: https://docs.doubleml.org/

## License

MIT © 2026 Fang Sun
