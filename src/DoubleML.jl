"""
    DoubleML

Julia implementation of Double/Debiased Machine Learning (Chernozhukov et al., 2018),
API-aligned with the Python package [DoubleML](https://docs.doubleml.org/).

# Models
- [`DoubleMLPLR`](@ref) — partially linear regression
- [`DoubleMLIRM`](@ref) — interactive regression model (binary treatment ATE)
- [`DoubleMLPLIV`](@ref) — partially linear IV regression
- [`DoubleMLIIVM`](@ref) — interactive IV model (binary D, binary Z → LATE)

# Learners
Duck-typed nuisance learners: any object supporting `fit!`, `predict` / `predict_proba`,
and `clone`. Built-ins use long-lived Julia packages (`DecisionTree.jl` random forests,
closed-form ridge/logistic).

# Example
```julia
using DoubleML
data = make_plr_data(n_obs=500, dim_x=20, theta=0.5; seed=3141)
ml_l = RidgeLearner(α=1.0)
ml_m = RidgeLearner(α=1.0)
dml = DoubleMLPLR(data, ml_l, ml_m; n_folds=5)
fit!(dml)
println(summary_table(dml))
```

See also: Python DoubleML (https://github.com/DoubleML/doubleml-for-py).
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
    # models
    AbstractDoubleML,
    DoubleMLPLR,
    DoubleMLIRM,
    DoubleMLPLIV,
    DoubleMLIIVM,
    confint,
    summary_table,
    dml_summary,
    # datasets
    make_plr_data,
    make_irm_data,
    make_pliv_data,
    make_iivm_data

include("learners.jl")
include("data.jl")
include("sample_splitting.jl")
include("base.jl")
include("plr.jl")
include("irm.jl")
include("pliv.jl")
include("iivm.jl")
include("datasets.jl")

end # module
