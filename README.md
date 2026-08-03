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
| **GATE / CATE** | Best linear predictor (`gate` / `cate` → `DoubleMLBLP`) for PLR & IRM | ✅ |
| **Policy tree** | Weighted classification of IRM score (`policy_tree` → `DoubleMLPolicyTree`) | ✅ |
| **PQ / QTE** | Potential quantiles & QTE (`score="PQ"`) | ✅ |
| **LPQ / LQTE** | Local (complier) potential quantiles (`score="LPQ"`, needs Z) | ✅ |
| **CVaR / CVaR-TE** | Conditional value at risk of potential outcomes (`score="CVaR"`) | ✅ |
| **APO / APOS** | Average potential outcomes + causal contrasts | ✅ |
| **DID** | Two-period ATT (`observational` / `experimental`) | ✅ |
| **DID multi (CS)** | Callaway–Sant’Anna: never/not-yet, anticipation, aggregate group/time/eventstudy, unit-IF joint SE, bootstrap, `p_adjust` | ✅ |
| **DIDCS** | Repeated cross-section two-period DiD (`observational` / `experimental`) | ✅ |
| **LPLR** | Logistic partially linear (binary Y) | ✅ |
| **PLPR** | Panel PLR: `fd_exact` / `wg_approx` / `cre_general` / `cre_normal` | ✅ |
| **RDFlex / RDD** | Iterative bandwidth, `fs_specification`, sharp/fuzzy local linear | ✅ |
| **SSM nonignorable** | Nested cross-fitting with Z (Python parity) | ✅ |
| **set_ml_nuisance_params!** | Per-learner hyperparams before `fit!` | ✅ |
| **store_models** | Nested CF models stored for SSM | ✅ |
| **effects_table / plot_effects** | DID multi event-study / group / time table | ✅ |
| **store_models (IRM/PLIV/SSM)** | Fold-level fitted nuisances when requested | ✅ |
| **tune! SSM/DID/PLPR** | Grid/random hyperparameter search | ✅ |
| **sensitivity_contour / sensitivity_plot** | Numerical cf_y × cf_d grid (plot-ready DataFrame; plot alias) | ✅ |
| **Framework sensitivity** | `sensitivity_analysis!` / contour / summary on `construct_framework` (PLR/IRM) | ✅ |
| **PSProcessor** | Propensity clip config (Python `PSProcessor` / `PSProcessorConfig`) on IRM | ✅ |
| **Confounded / hetero DGPs** | `make_confounded_plr_data`, `make_confounded_irm_data`, `make_heterogeneous_data`, discrete treatments | ✅ |
| **SSM** | Sample selection (`missing-at-random` / basic `nonignorable`) | ✅ |
| **RDD** | Sharp/fuzzy RD with ML residualization + local linear | ✅ |
| **Cluster SE** | One-way post-hoc + **cluster-in-fit** (1/2-way) for PLR/IRM | ✅ |
| **Multiple testing** | `p_adjust` Holm / Bonferroni / Romano–Wolf | ✅ |
| **IRM weights** | Observation weights for IRM ATE/ATTE | ✅ |
| **Framework** | `construct_framework` / `concat` / `+` `-` `*` joint IF algebra | ✅ |
| **Multi-treatment** | `d_cols` multi-column D for PLR/IRM (`use_other_treat_as_covariate`) | ✅ |
| **external_predictions** | Inject nuisance preds in `fit!` (PLR/IRM/IIVM) | ✅ |
| **evaluate_learners** | Cross-fit RMSE of stored nuisance predictions | ✅ |
| **IIVM subgroups / IPW** | `always_takers` / `never_takers` / `normalize_ipw` | ✅ |
| **DID multi experimental** | `score="experimental"` (no propensity) | ✅ |
| **Cluster PLIV/IIVM** | cluster-in-fit SE path | ✅ |

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
│   ├── framework.jl     # DoubleMLFramework / concat / algebra
│   ├── plr.jl
│   ├── irm.jl
│   ├── pliv.jl          # partially linear IV
│   ├── iivm.jl          # interactive IV (LATE)
│   ├── tune.jl          # grid / random hyperparameter search
│   ├── sensitivity.jl   # OVB sensitivity (Chernozhukov et al. 2022)
│   ├── blp.jl           # BLP / CATE / GATE
│   ├── policy_tree.jl   # optimal treatment policy trees (IRM)
│   ├── pq.jl            # potential quantiles (nonlinear DML)
│   ├── lpq.jl           # local potential quantiles (IV)
│   ├── cvar.jl          # CVaR of potential outcomes
│   ├── qte.jl           # QTE / LQTE / CVaR-TE
│   ├── apo.jl           # APO / APOS
│   ├── did.jl           # two-period DiD
│   ├── did_cs.jl        # repeated cross-section DiD
│   ├── did_multi.jl     # staggered DiD multi (CS toolbox)
│   ├── lplr.jl          # logistic PLR
│   ├── plpr.jl          # panel PLR (first-difference)
│   ├── ssm.jl           # sample selection
│   ├── rdd.jl           # regression discontinuity
│   ├── cluster.jl       # cluster SE + var_est
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

**GATE / CATE** (heterogeneous effects via best linear predictor; PLR & IRM):

```julia
fit!(dml)   # PLR or IRM (ATE)

# Group average treatment effects
groups = rand(["A", "B", "C"], size(data.x, 1))   # or an n×K dummy matrix
g = gate(dml, groups)
summary_table(g)
confint(g)                       # pointwise
confint(g; joint=true)           # simultaneous (Gaussian multiplier)

# Conditional ATE: project orthogonal signal onto a basis φ(x)
Φ = poly_basis(data.x[:, 1]; degree=3)   # or your own n×d design (e.g. B-splines)
c = cate(dml, Φ)
summary_table(c)
confint(c; basis=Φ)              # effect curve + CI at each row of Φ
```

**Policy tree** (optimal binary treatment rule from IRM ATE scores):

```julia
fit!(irm)   # DoubleMLIRM with score="ATE"
pt = policy_tree(irm, data.x[:, 1:3]; depth=2, min_samples_leaf=8)
summary_table(pt)
print_policy_tree(pt)
π = predict_policy(pt, data.x[:, 1:3])   # Vector{Int} in {0,1}
policy_value(pt)                         # (1/n) Σ (2π−1) ψ̂_b
```

**Distributional effects** (PQ / LPQ / CVaR and their treatment-effect wrappers):

```julia
# Potential quantile of Y(1) at τ=0.5  (IRM)
pq = DoubleMLPQ(data_irm, clf_g, clf_m; treatment=1, quantile=0.5)
fit!(pq)

# CVaR of Y(1)  (ml_g = regressor)
cvar = DoubleMLCVAR(data_irm, RidgeLearner(α=0.5), clf_m; treatment=1, quantile=0.5)
fit!(cvar)

# Local potential quantile for compliers  (needs instrument Z)
lpq = DoubleMLLPQ(data_iivm, clf_g, clf_m; treatment=1, quantile=0.5)
fit!(lpq)

# Treatment effects via DoubleMLQTE
qte  = DoubleMLQTE(data_irm,  clf_g, clf_m; quantiles=[0.25,0.5,0.75], score="PQ")
lqte = DoubleMLQTE(data_iivm, clf_g, clf_m; quantiles=[0.5], score="LPQ")
cte  = DoubleMLQTE(data_irm,  RidgeLearner(α=0.5), clf_m; quantiles=[0.5], score="CVaR")
fit!(qte); fit!(lqte); fit!(cte)
summary_table(qte)
# underlying models: *.modellist_0 / *.modellist_1
```

**APO / APOS** and **two-period DiD**:

```julia
# Average potential outcomes
apos = DoubleMLAPOS(data_irm, RidgeLearner(α=0.5), clf_m, [0.0, 1.0]; n_folds=5)
fit!(apos)
summary_table(apos)
causal_contrast(apos, 0.0)   # ATE ≈ APO(1) − APO(0)

# Two-period DiD (y = Y_post − Y_pre)
data_did = make_did_data(n_obs=800, theta=-2.0; seed=1)
did = DoubleMLDID(data_did, RidgeLearner(α=0.5), clf_m; score="observational")
fit!(did)
summary_table(did)

# Multi-period Callaway–Sant’Anna (panel long: id, t, d=first treatment period)
panel = make_did_panel_data(n_id=200, n_t=4, theta=2.0; seed=1)
cs = DoubleMLDIDMulti(
    panel, RidgeLearner(α=0.5), clf_m;
    control_group="never_treated",   # or "not_yet_treated"
    gt_combinations=:standard,       # :all | :universal
    anticipation_periods=0,
)
fit!(cs)
att_table(cs)                        # g, t_pre, t_eval, event_time, coef, se
aggregate(cs, :group)                # also :time, :eventstudy (unit-IF joint SE)
summary_table(aggregate(cs, :eventstudy))
bootstrap!(cs; n_rep_boot=500)
confint(cs; joint=true)
p_adjust(cs; method=:romano_wolf)    # :holm, :bonferroni

# Repeated cross-section DiD
rcs = make_did_cs_data(n_obs=1000, theta=-2.0; seed=2)
didcs = DoubleMLDIDCS(rcs, RidgeLearner(α=0.5), clf_m; score="observational")
fit!(didcs)

# Panel PLR — fd_exact | wg_approx | cre_general | cre_normal
panel_plpr = make_plpr_data(n_id=200, n_t=4, theta=0.5; seed=3)
plpr = DoubleMLPLPR(panel_plpr, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                    approach="cre_general")  # or fd_exact / wg_approx / cre_normal
fit!(plpr)

# RDFlex (alias of DoubleMLRDD): iterative bandwidth + fs_specification
rdd = RDFlex(make_rdd_data(n_obs=2000, tau=1.0; seed=6), RidgeLearner(α=0.5);
             n_iterations=2, fs_specification="cutoff and score")
fit!(rdd)

# Cluster-robust SE (any fitted model with psi stored)
cluster = rand(1:50, length(data.y))
cluster_se(dml; cluster=cluster)
apply_cluster_se!(dml; cluster=cluster)

# Framework: joint IF algebra (Python DoubleMLFramework)
f1 = construct_framework(dml)
f2 = construct_framework(irm)
fc = concat(f1, f2)
bootstrap!(fc; n_rep_boot=200)
confint(fc; joint=true)
contrast = f2 - f1          # or 2*f1, f1+f2

# Multi-treatment PLR
multi = make_plr_multi_data(n_obs=800, theta=[0.5, -0.3]; seed=4)
plr_m = DoubleMLPLR(multi, RidgeLearner(α=0.5), RidgeLearner(α=0.5))
fit!(plr_m)                 # coef length 2

# Cluster-in-fit (splits never split clusters)
cdata = make_plr_cluster_data(n_obs=600, n_clusters=40, theta=0.5; seed=5)
plr_c = DoubleMLPLR(cdata, RidgeLearner(α=0.5), RidgeLearner(α=0.5); n_folds=3)
fit!(plr_c)                 # SE uses cluster formula
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
