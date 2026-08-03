#!/usr/bin/env julia
# Demo: DoubleML.jl — PLR, IRM, PLIV, IIVM (API aligned with Python DoubleML)
#
#   julia --project=. examples/plr_irm_demo.jl

using DoubleML
using Random

println("="^60)
println("DoubleML.jl demo  |  Julia ", VERSION)
println("="^60)

θ = 0.5

# ---------- PLR ----------
data_plr = make_plr_data(n_obs=500, dim_x=20, theta=θ; seed=3141)
plr = DoubleMLPLR(data_plr, RidgeLearner(α=1.0), RidgeLearner(α=1.0); n_folds=5)
fit!(plr)
println("\n## PLR (partially linear regression)")
println("true θ = ", θ)
println(summary_table(plr))

# ---------- IRM ----------
data_irm = make_irm_data(n_obs=800, dim_x=10, theta=θ; seed=42)
irm = DoubleMLIRM(
    data_irm,
    RandomForestRegressorLearner(n_trees=80, max_depth=6, rng=MersenneTwister(1)),
    LogisticRegressionLearner(α=1.0);
    n_folds=5,
    trimming_threshold=0.01,
    rng=MersenneTwister(2),
)
fit!(irm)
println("\n## IRM (ATE)")
println("true ATE = ", θ)
println(summary_table(irm))

# ---------- PLIV ----------
data_pliv = make_pliv_data(n_obs=800, dim_x=15, dim_z=1, theta=θ; seed=99)
ml = RidgeLearner(α=0.5)
pliv = DoubleMLPLIV(data_pliv, clone(ml), clone(ml), clone(ml); n_folds=5, rng=MersenneTwister(99))
fit!(pliv)
println("\n## PLIV (partially linear IV)")
println("true θ = ", θ)
println(summary_table(pliv))

# ---------- IIVM / LATE ----------
data_iivm = make_iivm_data(n_obs=1500, dim_x=8, theta=θ; seed=7)
iivm = DoubleMLIIVM(
    data_iivm,
    RidgeLearner(α=0.5),
    LogisticRegressionLearner(α=0.5),
    LogisticRegressionLearner(α=0.5);
    n_folds=5,
    trimming_threshold=0.05,
    rng=MersenneTwister(7),
)
fit!(iivm)
println("\n## IIVM (LATE)")
println("true LATE ≈ ", θ, " (DGP simplified)")
println(summary_table(iivm))

# ---------- PLIV partialZ / partialXZ ----------
pliv_z = DoubleMLPLIV_partialZ(data_pliv, RidgeLearner(α=0.5); n_folds=5, rng=MersenneTwister(3))
fit!(pliv_z)
println("\n## PLIV partialZ")
println(summary_table(pliv_z))

data_pliv2 = make_pliv_data(n_obs=800, dim_x=10, dim_z=2, theta=θ; seed=5)
ml2 = RidgeLearner(α=0.5)
pliv_xz = DoubleMLPLIV_partialXZ(data_pliv2, clone(ml2), clone(ml2), clone(ml2);
                                 n_folds=5, rng=MersenneTwister(5))
fit!(pliv_xz)
println("\n## PLIV partialXZ")
println(summary_table(pliv_xz))

# ---------- Multiplier bootstrap ----------
bootstrap!(plr; method="normal", n_rep_boot=300, rng=MersenneTwister(9))
println("\n## Bootstrap joint CI (PLR)")
println(confint(plr; joint=true))

println("\nDone.")
