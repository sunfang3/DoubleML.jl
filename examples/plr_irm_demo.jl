#!/usr/bin/env julia
# Demo: DoubleML.jl — PLR, IRM, PLIV, IIVM (API aligned with Python DoubleML)
#
#   julia --project=. examples/plr_irm_demo.jl

using DoubleML
using Random
using Statistics

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

# ---------- Hyperparameter tuning ----------
data_t = make_plr_data(n_obs=600, dim_x=10, theta=θ; seed=101)
dml_t = DoubleMLPLR(data_t, RidgeLearner(α=50.0), RidgeLearner(α=50.0);
                    n_folds=5, rng=MersenneTwister(101))
tres = tune!(dml_t; param_grids=Dict(
    :ml_l => Dict(:α => [0.01, 0.1, 1.0, 10.0, 50.0]),
    :ml_m => Dict(:α => [0.01, 0.1, 1.0, 10.0, 50.0]),
), n_folds_tune=4, rng=MersenneTwister(102))
fit!(dml_t)
println("\n## PLR after tune!")
println("tuned ml_l: ", tres[:ml_l])
println("tuned ml_m: ", tres[:ml_m])
println(summary_table(dml_t))

# ---------- Sensitivity analysis ----------
sensitivity_analysis!(plr; cf_y=0.04, cf_d=0.03, rho=1.0, null_hypothesis=0.0)
println("\n## Sensitivity analysis (PLR)")
println(sensitivity_summary(plr))

# ---------- GATE / CATE (heterogeneous effects) ----------
println("\n## GATE (PLR) — random groups")
groups = rand(MersenneTwister(7), ["low", "mid", "high"], size(data_plr.x, 1))
gobj = gate(plr, groups)
println(summary_table(gobj))
println(confint(gobj; joint=true, n_rep_boot=300, rng=MersenneTwister(8)))

println("\n## CATE (IRM) — polynomial in X1")
Phi = poly_basis(data_irm.x[:, 1]; degree=2)
cobj = cate(irm, Phi)
println(summary_table(cobj))
# effect curve on a grid
xg = collect(range(extrema(data_irm.x[:, 1])...; length=8))
println(confint(cobj; basis=poly_basis(xg; degree=2)))

# ---------- Policy tree (IRM) ----------
println("\n## Policy tree (IRM) — depth 2 on X[:, 1:3]")
pt = policy_tree(irm, data_irm.x[:, 1:3]; depth=2, min_samples_leaf=15,
                 rng=MersenneTwister(9))
println(summary_table(pt))
println("policy_value = ", policy_value(pt))
print_policy_tree(pt)
pi_hat = predict_policy(pt, data_irm.x[:, 1:3])
println("share treated by policy: ", round(mean(pi_hat); digits=3))

# ---------- Potential quantiles / QTE ----------
println("\n## Potential quantile PQ (treatment=1, τ=0.5)")
pq = DoubleMLPQ(
    data_irm,
    LogisticRegressionLearner(α=0.5),
    LogisticRegressionLearner(α=0.5);
    treatment=1, quantile=0.5, n_folds=3,
    trimming_threshold=0.05, rng=MersenneTwister(11),
)
fit!(pq)
println(summary_table(pq))

println("\n## QTE at τ ∈ {0.25, 0.5, 0.75}")
qte = DoubleMLQTE(
    data_irm,
    LogisticRegressionLearner(α=0.5),
    LogisticRegressionLearner(α=0.5);
    quantiles=[0.25, 0.5, 0.75], n_folds=3,
    trimming_threshold=0.05, rng=MersenneTwister(12),
)
fit!(qte)
println(summary_table(qte))

println("\n## CVaR (treatment=1, τ=0.5) and CVaR-TE")
cvar = DoubleMLCVAR(
    data_irm, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
    treatment=1, quantile=0.5, n_folds=3,
    trimming_threshold=0.05, rng=MersenneTwister(13),
)
fit!(cvar)
println(summary_table(cvar))
cte = DoubleMLQTE(
    data_irm, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
    quantiles=[0.5], score="CVaR", n_folds=3,
    trimming_threshold=0.05, rng=MersenneTwister(14),
)
fit!(cte)
println(summary_table(cte))

println("\n## LPQ / LQTE (IIVM data with instrument Z)")
lpq = DoubleMLLPQ(
    data_iivm, LogisticRegressionLearner(α=0.5), LogisticRegressionLearner(α=0.5);
    treatment=1, quantile=0.5, n_folds=3,
    trimming_threshold=0.05, rng=MersenneTwister(15),
)
fit!(lpq)
println(summary_table(lpq))
lqte = DoubleMLQTE(
    data_iivm, LogisticRegressionLearner(α=0.5), LogisticRegressionLearner(α=0.5);
    quantiles=[0.5], score="LPQ", n_folds=3,
    trimming_threshold=0.05, rng=MersenneTwister(16),
)
fit!(lqte)
println(summary_table(lqte))

println("\nDone.")
