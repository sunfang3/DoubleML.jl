#!/usr/bin/env julia
# Demo: DoubleML.jl PLR & IRM — API aligned with Python DoubleML
#
#   julia --project=. examples/plr_irm_demo.jl

using DoubleML
using Random

println("="^60)
println("DoubleML.jl demo  |  Julia ", VERSION)
println("="^60)

# ---------- PLR ----------
Random.seed!(3141)
θ = 0.5
data_plr = make_plr_data(n_obs=500, dim_x=20, theta=θ; seed=3141)

ml_l = RidgeLearner(α=1.0)
ml_m = RidgeLearner(α=1.0)
plr = DoubleMLPLR(data_plr, ml_l, ml_m; n_folds=5)
fit!(plr)

println("\n## Partially Linear Regression (PLR)")
println("true θ = ", θ)
println(summary_table(plr))

# ---------- IRM ----------
Random.seed!(42)
data_irm = make_irm_data(n_obs=800, dim_x=10, theta=θ; seed=42)
irm = DoubleMLIRM(
    data_irm,
    RandomForestRegressorLearner(n_trees=100, max_depth=6, rng=MersenneTwister(1)),
    LogisticRegressionLearner(α=1.0);
    n_folds=5,
    trimming_threshold=0.01,
    rng=MersenneTwister(2),
)
fit!(irm)

println("\n## Interactive Regression Model (IRM / ATE)")
println("true ATE = ", θ)
println(summary_table(irm))

println("\nDone.")
