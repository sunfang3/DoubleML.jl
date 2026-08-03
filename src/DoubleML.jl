"""
    DoubleML

Julia implementation of Double/Debiased Machine Learning (Chernozhukov et al., 2018),
API-aligned with the Python package [DoubleML](https://docs.doubleml.org/).

# Models
- [`DoubleMLPLR`](@ref) — partially linear regression
- [`DoubleMLIRM`](@ref) — interactive regression model (binary treatment ATE)
- [`DoubleMLPLIV`](@ref) — partially linear IV (`:partialX` / `:partialZ` / `:partialXZ`)
- [`DoubleMLIIVM`](@ref) — interactive IV model (binary D, binary Z → LATE)
- [`DoubleMLPQ`](@ref) — potential quantiles (binary treatment)
- [`DoubleMLLPQ`](@ref) — local potential quantiles (IV / compliers)
- [`DoubleMLCVAR`](@ref) — CVaR of potential outcomes
- [`DoubleMLQTE`](@ref) — QTE / LQTE / CVaR-TE (`score="PQ"|"LPQ"|"CVaR"`)
- [`DoubleMLAPO`](@ref) / [`DoubleMLAPOS`](@ref) — average potential outcomes
- [`DoubleMLDID`](@ref) — two-period difference-in-differences ATT
- [`DoubleMLDIDCS`](@ref) — repeated cross-section DiD (two periods)
- [`DoubleMLDIDMulti`](@ref) — staggered group–time ATTs (Callaway–Sant’Anna)
- [`DoubleMLLPLR`](@ref) — logistic partially linear regression
- [`DoubleMLPLPR`](@ref) — partially linear panel regression (first-difference)
- [`DoubleMLSSM`](@ref) — sample selection model
- [`DoubleMLRDD`](@ref) — regression discontinuity (sharp/fuzzy)

# Inference
- Pointwise: `confint(m)`
- Joint (multiplier bootstrap): `bootstrap!(m)` then `confint(m; joint=true)`
- Multiple testing: `p_adjust(m; method=:holm|:bonferroni|:romano_wolf)`
- Cluster SE: `cluster_se` / `apply_cluster_se!`
- Sensitivity (omitted confounders): `sensitivity_analysis!` (PLR / IRM)
- Heterogeneous effects: `gate` / `cate` → [`DoubleMLBLP`](@ref) (PLR / IRM)
- Policy learning: `policy_tree` → [`DoubleMLPolicyTree`](@ref) (IRM ATE)

# Tuning
- `tune!(m; param_grids=...)` then `fit!(m)`

See also: https://github.com/DoubleML/doubleml-for-py
"""
module DoubleML

using LinearAlgebra
using Random
using Statistics
using StatsBase
using Distributions
using DataFrames
using Tables
using DecisionTree

export
    # data
    DoubleMLData,
    n_instr,
    # learners
    AbstractLearner,
    RidgeLearner,
    LinearRegressionLearner,
    LogisticRegressionLearner,
    RandomForestRegressorLearner,
    RandomForestClassifierLearner,
    fit!,
    predict,
    predict_proba,
    clone,
    get_params,
    set_params!,
    # models
    AbstractDoubleML,
    DoubleMLPLR,
    DoubleMLIRM,
    DoubleMLPLIV,
    DoubleMLPLIV_partialZ,
    DoubleMLPLIV_partialXZ,
    DoubleMLIIVM,
    DoubleMLPQ,
    DoubleMLLPQ,
    DoubleMLCVAR,
    DoubleMLQTE,
    DoubleMLAPO,
    DoubleMLAPOS,
    causal_contrast,
    DoubleMLDID,
    DoubleMLDIDCS,
    DoubleMLDIDMulti,
    DoubleMLPanelData,
    att_table,
    aggregate,
    DIDAggregation,
    DoubleMLLPLR,
    DoubleMLPLPR,
    DoubleMLSSM,
    DoubleMLRDD,
    confint,
    summary_table,
    dml_summary,
    bootstrap!,
    BootstrapResult,
    p_adjust,
    set_sample_splitting!,
    cluster_se,
    apply_cluster_se!,
    # framework
    DoubleMLCore,
    DoubleMLFramework,
    construct_framework,
    framework,
    concat,
    n_treat,
    design_for_treatment,
    is_cluster_data,
    n_cluster_vars,
    # tuning
    tune!,
    tune_learner,
    TuneResult,
    cv_score,
    # sensitivity
    sensitivity_analysis!,
    sensitivity_summary,
    sensitivity_benchmark,
    SensitivityResult,
    SensitivityElements,
    # heterogeneous effects (CATE / GATE)
    DoubleMLBLP,
    cate,
    gate,
    group_dummies,
    poly_basis,
    # policy learning
    DoubleMLPolicyTree,
    policy_tree,
    predict_policy,
    policy_value,
    print_policy_tree,
    # datasets
    make_plr_data,
    make_plr_multi_data,
    make_plr_cluster_data,
    make_irm_data,
    make_pliv_data,
    make_iivm_data,
    make_did_data,
    make_did_cs_data,
    make_lplr_data,
    make_plpr_data,
    make_ssm_data,
    make_did_panel_data,
    make_rdd_data

include("learners.jl")
include("data.jl")
include("sample_splitting.jl")
include("base.jl")       # AbstractDoubleML, confint, summary_table, p_adjust
include("bootstrap.jl")  # BootstrapResult, bootstrap! (needs AbstractDoubleML)
include("framework.jl")  # DoubleMLFramework / concat / algebra
include("cluster.jl")    # cluster SE + var_est
include("sensitivity.jl")
include("plr.jl")
include("irm.jl")
include("pliv.jl")
include("iivm.jl")
include("blp.jl")        # BLP / cate / gate (needs PLR & IRM)
include("policy_tree.jl")  # policy learning (needs IRM + orth signal)
include("pq.jl")         # potential quantiles (nonlinear DML)
include("lpq.jl")        # local potential quantiles (IV)
include("cvar.jl")       # CVaR of potential outcomes
include("qte.jl")        # QTE / LQTE / CVaR-TE
include("apo.jl")        # APO / APOS
include("did.jl")        # two-period DiD
include("did_cs.jl")     # repeated cross-section DiD
include("did_multi.jl")  # staggered DiD multi (+ unit IF bootstrap)
include("lplr.jl")       # logistic PLR
include("plpr.jl")       # panel PLR (first-difference)
include("ssm.jl")        # sample selection
include("rdd.jl")        # regression discontinuity
include("tune.jl")
include("datasets.jl")

end # module
