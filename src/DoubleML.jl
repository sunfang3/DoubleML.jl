"""
    DoubleML

Julia implementation of Double/Debiased Machine Learning (Chernozhukov et al., 2018),
API-aligned with the Python package [DoubleML](https://docs.doubleml.org/).

# Models
- [`DoubleMLPLR`](@ref) — partially linear regression
- [`DoubleMLIRM`](@ref) — interactive regression model (binary treatment ATE)
- [`DoubleMLPLIV`](@ref) — partially linear IV (`:partialX` / `:partialZ` / `:partialXZ`)
- [`DoubleMLIIVM`](@ref) — interactive IV model (binary D, binary Z → LATE)

# Inference
- Pointwise: `confint(m)`
- Joint (multiplier bootstrap): `bootstrap!(m)` then `confint(m; joint=true)`
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
    confint,
    summary_table,
    dml_summary,
    bootstrap!,
    BootstrapResult,
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
    make_irm_data,
    make_pliv_data,
    make_iivm_data

include("learners.jl")
include("data.jl")
include("sample_splitting.jl")
include("base.jl")       # AbstractDoubleML, confint, summary_table
include("bootstrap.jl")  # BootstrapResult, bootstrap! (needs AbstractDoubleML)
include("sensitivity.jl")
include("plr.jl")
include("irm.jl")
include("pliv.jl")
include("iivm.jl")
include("blp.jl")        # BLP / cate / gate (needs PLR & IRM)
include("policy_tree.jl")  # policy learning (needs IRM + orth signal)
include("tune.jl")
include("datasets.jl")

end # module
