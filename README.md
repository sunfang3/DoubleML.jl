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
| **PLIV** | `partialling out` (1 or multi Z) | ✅ |
| **PLIV** | `IV-type` (single Z) | ✅ |
| **IIVM** | `LATE` | ✅ |
| Multiplier bootstrap | — | ⏳ planned |
| Hyperparameter tuning | — | ⏳ planned |

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
│   └── datasets.jl
├── test/runtests.jl
├── examples/plr_irm_demo.jl
└── README.md
```

## IV models in brief

**PLIV** — continuous treatment, instrument(s) `Z` in `DoubleMLData`:

```julia
data = make_pliv_data(n_obs=1000, dim_z=1, theta=0.5; seed=1)
# or from a DataFrame:
# DoubleMLData(df; y_col="y", d_cols="d", z_cols="Z1")
DoubleMLPLIV(data, ml_l, ml_m, ml_r)  # E[Y|X], E[Z|X], E[D|X]
```

**IIVM** — binary `D` and binary `Z`, target = LATE:

```julia
data = make_iivm_data(n_obs=2000, theta=0.5; seed=1)
DoubleMLIIVM(data, ml_g, ml_m, ml_r)  # E[Y|X,Z], E[Z|X], E[D|X,Z]
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
