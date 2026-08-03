using Test
using Random
using DoubleML
using DataFrames
using Statistics

@testset "DoubleML.jl" begin

    @testset "DoubleMLData" begin
        n, p = 100, 5
        X = randn(n, p)
        y = randn(n)
        d = randn(n)
        data = DoubleMLData(X, y, d)
        @test size(data.x) == (n, p)
        @test length(data.y) == n

        df = DataFrame(X, :auto)
        df.y = y
        df.d = d
        data2 = DoubleMLData(df; y_col="y", d_cols="d")
        @test size(data2.x, 2) == p
    end

    @testset "Learners" begin
        Random.seed!(1)
        n, p = 200, 3
        X = randn(n, p)
        β = [1.0, -0.5, 0.2]
        y = X * β + 0.1 * randn(n)

        ridge = RidgeLearner(α=0.1)
        fit!(ridge, X, y)
        yhat = predict(ridge, X)
        @test length(yhat) == n
        @test cor(yhat, y) > 0.9

        # classification
        logits = X * β
        d = Float64.(logits .> 0)
        clf = LogisticRegressionLearner(α=0.1)
        fit!(clf, X, d)
        p_hat = predict_proba(clf, X)
        @test all(0 .<= p_hat .<= 1)
        @test mean((p_hat .> 0.5) .== d) > 0.7
    end

    @testset "PLR recovers theta" begin
        Random.seed!(3141)
        θ_true = 0.5
        data = make_plr_data(n_obs=800, dim_x=10, theta=θ_true; seed=3141)
        dml = DoubleMLPLR(data, RidgeLearner(α=1.0), RidgeLearner(α=1.0); n_folds=5)
        fit!(dml)
        @test abs(dml.coef[1] - θ_true) < 0.15
        @test dml.se[1] > 0
        s = summary_table(dml)
        @test s.coef[1] ≈ dml.coef[1]
        ci = confint(dml)
        @test ci.lower[1] < dml.coef[1] < ci.upper[1]
        # true theta inside 95% CI with high probability
        @test ci.lower[1] - 0.1 < θ_true < ci.upper[1] + 0.1
    end

    @testset "IRM recovers ATE" begin
        Random.seed!(42)
        θ_true = 0.5
        data = make_irm_data(n_obs=1000, dim_x=5, theta=θ_true; seed=42)
        dml = DoubleMLIRM(
            data,
            RidgeLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            n_folds=5,
            trimming_threshold=0.01,
        )
        fit!(dml)
        @test abs(dml.coef[1] - θ_true) < 0.2
        @test dml.se[1] > 0
    end

    @testset "PLR with RandomForest" begin
        data = make_plr_data(n_obs=400, dim_x=5, theta=0.5; seed=7)
        ml = RandomForestRegressorLearner(n_trees=50, max_depth=5, rng=MersenneTwister(7))
        dml = DoubleMLPLR(data, clone(ml), clone(ml); n_folds=3, rng=MersenneTwister(7))
        fit!(dml)
        @test isfinite(dml.coef[1])
        @test dml.se[1] > 0
    end

    @testset "PLIV recovers theta" begin
        θ_true = 0.5
        data = make_pliv_data(n_obs=1200, dim_x=8, dim_z=1, theta=θ_true; seed=99)
        @test n_instr(data) == 1
        ml = RidgeLearner(α=0.5)
        dml = DoubleMLPLIV(data, clone(ml), clone(ml), clone(ml); n_folds=5, rng=MersenneTwister(99))
        fit!(dml)
        @test abs(dml.coef[1] - θ_true) < 0.2
        @test dml.se[1] > 0
        s = summary_table(dml)
        @test s.coef[1] ≈ dml.coef[1]
    end

    @testset "PLIV multi-instrument" begin
        θ_true = 0.5
        data = make_pliv_data(n_obs=1500, dim_x=6, dim_z=3, theta=θ_true; seed=123)
        @test n_instr(data) == 3
        ml = RidgeLearner(α=0.5)
        dml = DoubleMLPLIV(data, clone(ml), clone(ml), clone(ml); n_folds=5, rng=MersenneTwister(123))
        fit!(dml)
        @test abs(dml.coef[1] - θ_true) < 0.25
        @test isfinite(dml.se[1])
    end

    @testset "PLIV IV-type score" begin
        data = make_pliv_data(n_obs=1000, dim_x=5, dim_z=1, theta=0.5; seed=7)
        ml = RidgeLearner(α=0.5)
        dml = DoubleMLPLIV(data, clone(ml), clone(ml), clone(ml);
                           ml_g=clone(ml), score="IV-type", n_folds=5, rng=MersenneTwister(7))
        fit!(dml)
        @test isfinite(dml.coef[1])
        @test dml.se[1] > 0
    end

    @testset "IIVM recovers LATE" begin
        θ_true = 0.5
        data = make_iivm_data(n_obs=2000, dim_x=5, theta=θ_true; seed=314)
        @test n_instr(data) == 1
        dml = DoubleMLIIVM(
            data,
            RidgeLearner(α=0.5),
            LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            n_folds=5,
            trimming_threshold=0.05,
            rng=MersenneTwister(314),
        )
        fit!(dml)
        # LATE recovery is noisier; allow a wider band
        @test abs(dml.coef[1] - θ_true) < 0.45
        @test dml.se[1] > 0
        @test isfinite(dml.coef[1])
    end

    @testset "DataFrame with instruments" begin
        df = make_pliv_data(n_obs=200, dim_x=3, dim_z=1, theta=0.5;
                            return_type=:DataFrame, seed=1)
        data = DoubleMLData(df; y_col="y", d_cols="d", z_cols="Z1")
        @test n_instr(data) == 1
        @test data.z_cols == ["Z1"]
    end

    @testset "Multiplier bootstrap + joint CI" begin
        data = make_plr_data(n_obs=600, dim_x=6, theta=0.5; seed=11)
        dml = DoubleMLPLR(data, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                          n_folds=4, rng=MersenneTwister(11))
        fit!(dml)
        bootstrap!(dml; method="normal", n_rep_boot=200, rng=MersenneTwister(12))
        @test dml.boot !== nothing
        @test size(dml.boot.boot_t_stat, 1) == 200
        ci_pt = confint(dml; joint=false)
        ci_jt = confint(dml; joint=true)
        # both CIs cover the point estimate
        @test ci_pt.lower[1] < dml.coef[1] < ci_pt.upper[1]
        @test ci_jt.lower[1] < dml.coef[1] < ci_jt.upper[1]
        @test ci_jt.joint[1] == true
        # finite-sample joint critical value ≈ normal; allow slight noise
        @test (ci_jt.upper[1] - ci_jt.lower[1]) >= 0.85 * (ci_pt.upper[1] - ci_pt.lower[1])
        # wild / Bayes weights also run
        bootstrap!(dml; method="wild", n_rep_boot=50, rng=MersenneTwister(1))
        bootstrap!(dml; method="Bayes", n_rep_boot=50, rng=MersenneTwister(2))
        @test dml.boot.method == "Bayes"
    end

    @testset "PLIV partialZ" begin
        # Use many instruments (as in DoubleML partialZ fixtures) so the
        # first-stage projection is informative without residualizing Y.
        data = make_pliv_data(n_obs=1500, dim_x=5, dim_z=20, theta=0.5; seed=21)
        dml = DoubleMLPLIV_partialZ(data, RidgeLearner(α=1.0);
                                   n_folds=4, rng=MersenneTwister(21))
        @test dml.partial_mode == :partialZ
        fit!(dml)
        @test isfinite(dml.coef[1])
        @test dml.se[1] > 0
        # partialZ without Y residualization is noisier — wide band
        @test abs(dml.coef[1] - 0.5) < 0.75
    end

    @testset "PLIV partialXZ" begin
        data = make_pliv_data(n_obs=1200, dim_x=5, dim_z=2, theta=0.5; seed=22)
        ml = RidgeLearner(α=0.5)
        dml = DoubleMLPLIV_partialXZ(data, clone(ml), clone(ml), clone(ml);
                                     n_folds=4, rng=MersenneTwister(22))
        @test dml.partial_mode == :partialXZ
        fit!(dml)
        @test isfinite(dml.coef[1])
        @test dml.se[1] > 0
        @test abs(dml.coef[1] - 0.5) < 0.35
    end

    @testset "Learner set_params / get_params" begin
        r = RidgeLearner(α=1.0)
        @test get_params(r)[:α] == 1.0
        set_params!(r; α=0.25)
        @test r.α == 0.25
        @test !r.fitted
        clf = LogisticRegressionLearner(α=2.0)
        set_params!(clf; α=0.5, max_iter=50)
        @test clf.α == 0.5 && clf.max_iter == 50
        rf = RandomForestRegressorLearner(n_trees=10, max_depth=3)
        set_params!(rf; n_trees=20, max_depth=5)
        @test rf.n_trees == 20 && rf.max_depth == 5
    end

    @testset "tune_learner grid search" begin
        Random.seed!(1)
        n, p = 300, 4
        X = randn(n, p)
        y = X[:, 1] .+ 0.1 .* randn(n)
        learner = RidgeLearner(α=100.0)  # deliberately poor default
        best, res = tune_learner(learner, X, y, Dict(:α => [0.01, 0.1, 1.0, 10.0, 100.0]);
                                 n_folds=4, rng=MersenneTwister(1))
        @test res.best_score < Inf
        @test length(res.all_scores) == 5
        # best α should prefer small regularization for this signal
        @test res.best_params[:α] <= 1.0
        @test get_params(best)[:α] == res.best_params[:α]
    end

    @testset "tune! PLR then fit" begin
        data = make_plr_data(n_obs=700, dim_x=8, theta=0.5; seed=33)
        dml = DoubleMLPLR(data, RidgeLearner(α=50.0), RidgeLearner(α=50.0);
                          n_folds=4, rng=MersenneTwister(33))
        tres = tune!(dml; param_grids=Dict(
            :ml_l => Dict(:α => [0.01, 0.1, 1.0, 10.0, 50.0]),
            :ml_m => Dict(:α => [0.01, 0.1, 1.0, 10.0, 50.0]),
        ), n_folds_tune=3, rng=MersenneTwister(34))
        @test haskey(tres, :ml_l) && haskey(tres, :ml_m)
        @test !dml.fitted
        fit!(dml)
        @test abs(dml.coef[1] - 0.5) < 0.2
        @test dml.se[1] > 0
    end

    @testset "tune! random search IRM" begin
        data = make_irm_data(n_obs=800, dim_x=5, theta=0.5; seed=44)
        dml = DoubleMLIRM(
            data,
            RidgeLearner(α=10.0),
            LogisticRegressionLearner(α=10.0);
            n_folds=4,
            trimming_threshold=0.05,
            rng=MersenneTwister(44),
        )
        tres = tune!(dml; param_grids=Dict(
            :ml_g => Dict(:α => [0.1, 1.0, 10.0]),
            :ml_m => Dict(:α => [0.1, 1.0, 10.0]),
        ), search_mode=:random, n_iter=4, n_folds_tune=3, rng=MersenneTwister(45))
        @test haskey(tres, :ml_g)
        fit!(dml)
        @test isfinite(dml.coef[1])
    end

    @testset "Sensitivity analysis PLR" begin
        data = make_plr_data(n_obs=800, dim_x=10, theta=0.5; seed=77)
        dml = DoubleMLPLR(data, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                          n_folds=5, rng=MersenneTwister(77))
        fit!(dml)
        @test dml.sens_elements !== nothing
        @test all(dml.sens_elements.sigma2 .> 0)
        @test all(dml.sens_elements.nu2 .> 0)

        # no confounding → bounds collapse to θ
        r0 = sensitivity_analysis!(dml; cf_y=0.0, cf_d=0.0, rho=1.0, level=0.95)
        @test r0.theta_lower[1] ≈ dml.coef[1] atol=1e-10
        @test r0.theta_upper[1] ≈ dml.coef[1] atol=1e-10

        r = sensitivity_analysis!(dml; cf_y=0.04, cf_d=0.03, rho=1.0, level=0.95,
                                  null_hypothesis=0.0)
        @test r.theta_lower[1] < dml.coef[1] < r.theta_upper[1]
        @test r.ci_lower[1] <= r.theta_lower[1]
        @test r.ci_upper[1] >= r.theta_upper[1]
        @test 0 <= r.rv[1] < 1
        @test 0 <= r.rva[1] < 1
        # stronger confounding → wider bounds
        r2 = sensitivity_analysis!(dml; cf_y=0.15, cf_d=0.15, rho=1.0)
        @test (r2.theta_upper[1] - r2.theta_lower[1]) >
              (r.theta_upper[1] - r.theta_lower[1]) - 1e-12
        s = sensitivity_summary(dml)
        @test occursin("Sensitivity Analysis", s)
        @test occursin("Robustness Values", s)
    end

    @testset "Sensitivity analysis IRM" begin
        data = make_irm_data(n_obs=1000, dim_x=5, theta=0.5; seed=88)
        dml = DoubleMLIRM(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=5, trimming_threshold=0.05, rng=MersenneTwister(88),
        )
        fit!(dml)
        r = sensitivity_analysis!(dml; cf_y=0.03, cf_d=0.03, rho=1.0)
        @test r.theta_lower[1] < dml.coef[1] < r.theta_upper[1]
        @test isfinite(r.rv[1]) && isfinite(r.rva[1])
    end

    @testset "Sensitivity benchmark" begin
        # long: all X; short: omit X1 (main confounder in make_plr_data)
        data_long = make_plr_data(n_obs=900, dim_x=6, theta=0.5; seed=91)
        dml_long = DoubleMLPLR(data_long, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                               n_folds=4, rng=MersenneTwister(91))
        fit!(dml_long)
        # short: drop first covariate
        Xshort = data_long.x[:, 2:end]
        data_short = DoubleMLData(Xshort, data_long.y, data_long.d)
        dml_short = DoubleMLPLR(data_short, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                                n_folds=4, rng=MersenneTwister(92))
        fit!(dml_short)
        bm = sensitivity_benchmark(dml_long, dml_short)
        @test 0 <= bm.cf_y < 1
        @test 0 <= bm.cf_d < 1
        @test isfinite(bm.delta_theta)
    end

    @testset "GATE PLR" begin
        # constant θ ⇒ all group effects ≈ θ
        data = make_plr_data(n_obs=1000, dim_x=8, theta=0.5; seed=101)
        dml = DoubleMLPLR(data, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                          n_folds=5, rng=MersenneTwister(101))
        fit!(dml)
        rng = MersenneTwister(102)
        groups = rand(rng, ["A", "B", "C"], size(data.x, 1))
        g = gate(dml, groups)
        @test g.fitted
        @test g.is_gate
        @test length(g.coef) == 3
        @test all(isfinite, g.coef)
        # all groups should recover ≈ 0.5
        @test all(abs.(g.coef .- 0.5) .< 0.25)
        ci = confint(g)
        @test nrow(ci) == 3
        @test all(ci.lower .<= ci.effect .<= ci.upper)
        st = summary_table(g)
        @test nrow(st) == 3
        # joint CI wider (or equal) than pointwise on average
        cij = confint(g; joint=true, n_rep_boot=200, rng=MersenneTwister(103))
        @test all(cij.joint)
        @test mean(cij.upper .- cij.lower) >= mean(ci.upper .- ci.lower) - 1e-8
    end

    @testset "GATE IRM" begin
        data = make_irm_data(n_obs=1200, dim_x=6, theta=0.5; seed=111)
        dml = DoubleMLIRM(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=5, trimming_threshold=0.05, rng=MersenneTwister(111),
        )
        fit!(dml)
        # dummy matrix input
        gvec = rand(MersenneTwister(112), 1:2, size(data.x, 1))
        G = Float64.(hcat(gvec .== 1, gvec .== 2))
        g = gate(dml, G)
        @test length(g.coef) == 2
        @test all(abs.(g.coef .- 0.5) .< 0.35)
        ci = confint(g)
        @test nrow(ci) == 2
    end

    @testset "CATE PLR poly basis" begin
        data = make_plr_data(n_obs=900, dim_x=5, theta=0.5; seed=121)
        dml = DoubleMLPLR(data, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                          n_folds=4, rng=MersenneTwister(121))
        fit!(dml)
        Φ = poly_basis(data.x[:, 1]; degree=2, include_intercept=true)
        c = cate(dml, Φ)
        @test c.fitted && !c.is_gate
        @test length(c.coef) == 3
        # predicted effects near constant 0.5
        ci = confint(c; basis=Φ)
        @test nrow(ci) == size(Φ, 1)
        @test mean(abs.(ci.effect .- 0.5)) < 0.3
        # coefficient-level confint
        ciβ = confint(c)
        @test nrow(ciβ) == 3
    end

    @testset "CATE IRM poly basis" begin
        data = make_irm_data(n_obs=1000, dim_x=5, theta=0.5; seed=131)
        dml = DoubleMLIRM(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=4, trimming_threshold=0.05, rng=MersenneTwister(131),
        )
        fit!(dml)
        Φ = poly_basis(data.x[:, 1]; degree=2)
        c = cate(dml, Φ)
        @test length(c.coef) == 3
        @test all(isfinite, c.coef)
        # evaluate on a short grid
        xg = range(minimum(data.x[:, 1]), maximum(data.x[:, 1]); length=20)
        Φg = poly_basis(collect(xg); degree=2)
        ci = confint(c; basis=Φg, joint=false)
        @test nrow(ci) == 20
        @test all(isfinite, ci.effect)
    end

    @testset "group_dummies helper" begin
        g = ["a", "b", "a", "c"]
        G, nm = group_dummies(g)
        @test size(G) == (4, 3)
        @test sum(G; dims=2) == ones(4, 1)  # exclusive
        @test nm == ["Group_a", "Group_b", "Group_c"]
    end

    @testset "Policy tree IRM" begin
        # DGP with heterogeneous sign: θ(x) = 1{X1 > 0} − 1{X1 ≤ 0}  (≈ ±1)
        # so optimal policy treats when X1 > 0
        rng = MersenneTwister(201)
        n, p = 1200, 4
        X = randn(rng, n, p)
        # propensity
        m0 = 1 ./ (1 .+ exp.(-0.5 .* X[:, 1]))
        d = Float64.(rand(rng, n) .< m0)
        tau = ifelse.(X[:, 1] .> 0, 1.0, -0.5)   # treat only when X1>0 is better
        g0 = X[:, 1] .+ X[:, 2]
        y = g0 .+ d .* tau .+ randn(rng, n)
        data = DoubleMLData(X, y, d)
        dml = DoubleMLIRM(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=5, trimming_threshold=0.05, rng=MersenneTwister(202),
        )
        fit!(dml)

        pt = policy_tree(dml, X[:, 1:3]; depth=2, min_samples_leaf=20,
                         rng=MersenneTwister(203))
        @test pt.fitted
        @test pt.depth == 2
        π = predict_policy(pt, X[:, 1:3])
        @test all(π .∈ Ref((0, 1)))
        @test 0 < mean(π) < 1

        # policy should prefer treating when X1 > 0
        rate_pos = mean(π[X[:, 1] .> 0])
        rate_neg = mean(π[X[:, 1] .<= 0])
        @test rate_pos > rate_neg + 0.15

        # policy value of learned policy ≥ always-control
        v_hat = policy_value(pt)
        v_never = policy_value(pt.orth_signal, zeros(Int, n))
        v_always = policy_value(pt.orth_signal, ones(Int, n))
        @test v_hat >= v_never - 1e-8
        # should also beat the worse of always/never
        @test v_hat >= min(v_never, v_always) - 1e-8

        st = summary_table(pt)
        @test nrow(st) == 1
        @test st.policy_value[1] ≈ v_hat atol=1e-12

        # DataFrame features + predict
        dfX = DataFrame(X[:, 1:3], ["f1", "f2", "f3"])
        pt2 = policy_tree(dml, dfX; depth=2, min_samples_leaf=20, rng=MersenneTwister(204))
        π2 = predict_policy(pt2, dfX)
        @test length(π2) == n
        @test all(π2 .∈ Ref((0, 1)))
    end

    @testset "Policy tree rejects non-ATE" begin
        data = make_irm_data(n_obs=400, dim_x=3, theta=0.5; seed=210)
        dml = DoubleMLIRM(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=3, score="ATTE", trimming_threshold=0.05, rng=MersenneTwister(210),
        )
        fit!(dml)
        @test_throws ArgumentError policy_tree(dml, data.x; depth=1)
    end

    @testset "Potential quantile PQ" begin
        # constant shift treatment: Y = g(X) + θ D + ε ⇒ Q_τ(1) − Q_τ(0) ≈ θ
        data = make_irm_data(n_obs=1500, dim_x=4, theta=0.5; seed=301)
        pq1 = DoubleMLPQ(
            data,
            LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            treatment=1, quantile=0.5, n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(301),
        )
        fit!(pq1)
        @test pq1.fitted
        @test isfinite(pq1.coef[1])
        @test pq1.se[1] > 0
        # median potential outcome treated should be finite and within y range
        @test minimum(data.y) <= pq1.coef[1] <= maximum(data.y)

        pq0 = DoubleMLPQ(
            data,
            LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            treatment=0, quantile=0.5, n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(302),
        )
        fit!(pq0)
        # QTE proxy
        @test isfinite(pq1.coef[1] - pq0.coef[1])
        st = summary_table(pq1)
        @test nrow(st) == 1
        ci = confint(pq1)
        @test ci.lower[1] <= pq1.coef[1] <= ci.upper[1]
    end

    @testset "QTE recovers location shift" begin
        # IRM DGP with additive θ ⇒ QTE(τ) ≈ θ for all τ
        data = make_irm_data(n_obs=1800, dim_x=4, theta=0.5; seed=311)
        qte = DoubleMLQTE(
            data,
            LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            quantiles=[0.5], n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(311),
        )
        fit!(qte)
        @test qte.fitted
        @test length(qte.coef) == 1
        # location shift recovery (allow moderate error for quantile + logistic nuisance)
        @test abs(qte.coef[1] - 0.5) < 0.45
        @test qte.se[1] > 0
        st = summary_table(qte)
        @test nrow(st) == 1
        # multi-quantile
        qte2 = DoubleMLQTE(
            data,
            LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            quantiles=[0.25, 0.5, 0.75], n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(312),
        )
        fit!(qte2)
        @test length(qte2.coef) == 3
        @test all(isfinite, qte2.coef)
        @test length(qte2.modellist_0) == 3
        @test length(qte2.modellist_1) == 3
    end

    @testset "CVaR potential outcome" begin
        data = make_irm_data(n_obs=1500, dim_x=4, theta=0.5; seed=401)
        c1 = DoubleMLCVAR(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            treatment=1, quantile=0.5, n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(401),
        )
        fit!(c1)
        @test c1.fitted
        @test isfinite(c1.coef[1])
        @test c1.se[1] > 0
        # CVaR ≥ PQ roughly for same τ (upper-tail mean)
        pq1 = DoubleMLPQ(
            data, LogisticRegressionLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            treatment=1, quantile=0.5, n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(402),
        )
        fit!(pq1)
        @test c1.coef[1] >= pq1.coef[1] - 0.5  # soft: estimation noise
        st = summary_table(c1)
        @test nrow(st) == 1
    end

    @testset "CVaR-TE via QTE score" begin
        data = make_irm_data(n_obs=1500, dim_x=4, theta=0.5; seed=411)
        qte = DoubleMLQTE(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            quantiles=[0.5], score="CVaR", n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(411),
        )
        fit!(qte)
        @test qte.score == "CVaR"
        @test isfinite(qte.coef[1])
        @test qte.se[1] > 0
        @test qte.modellist_0[1] isa DoubleMLCVAR
        # location-shift DGP → CVaR-TE also near θ
        @test abs(qte.coef[1] - 0.5) < 0.55
    end

    @testset "LPQ local potential quantile" begin
        data = make_iivm_data(n_obs=2500, dim_x=4, theta=0.5; seed=421)
        lpq = DoubleMLLPQ(
            data, LogisticRegressionLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            treatment=1, quantile=0.5, n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(421),
        )
        fit!(lpq)
        @test lpq.fitted
        @test isfinite(lpq.coef[1])
        @test lpq.se[1] > 0
        @test minimum(data.y) <= lpq.coef[1] <= maximum(data.y)
        st = summary_table(lpq)
        @test nrow(st) == 1
    end

    @testset "LQTE via QTE score=LPQ" begin
        data = make_iivm_data(n_obs=2500, dim_x=4, theta=0.5; seed=431)
        qte = DoubleMLQTE(
            data, LogisticRegressionLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            quantiles=[0.5], score="LPQ", n_folds=3,
            trimming_threshold=0.05, rng=MersenneTwister(431),
        )
        fit!(qte)
        @test qte.score == "LPQ"
        @test isfinite(qte.coef[1])
        @test qte.se[1] > 0
        @test qte.modellist_0[1] isa DoubleMLLPQ
        # additive LATE-style shift → rough recovery
        @test abs(qte.coef[1] - 0.5) < 0.7
    end

    @testset "QTE score validation" begin
        data = make_irm_data(n_obs=200, dim_x=3, theta=0.5; seed=441)
        @test_throws ArgumentError DoubleMLQTE(
            data, LogisticRegressionLearner(), LogisticRegressionLearner();
            score="LPQ",  # no instrument
        )
        @test_throws ArgumentError DoubleMLQTE(
            data, LogisticRegressionLearner(), LogisticRegressionLearner();
            score="FOO",
        )
    end

    @testset "APO and APOS" begin
        data = make_irm_data(n_obs=1200, dim_x=4, theta=0.5; seed=501)
        # APO(1) − APO(0) ≈ ATE ≈ 0.5
        apo1 = DoubleMLAPO(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            treatment_level=1, n_folds=3, trimming_threshold=0.05,
            rng=MersenneTwister(501),
        )
        apo0 = DoubleMLAPO(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            treatment_level=0, n_folds=3, trimming_threshold=0.05,
            rng=MersenneTwister(502),
        )
        fit!(apo1); fit!(apo0)
        @test isfinite(apo1.coef[1]) && isfinite(apo0.coef[1])
        @test abs((apo1.coef[1] - apo0.coef[1]) - 0.5) < 0.45

        apos = DoubleMLAPOS(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5),
            [0.0, 1.0];
            n_folds=3, trimming_threshold=0.05, rng=MersenneTwister(503),
        )
        fit!(apos)
        @test length(apos.coef) == 2
        ct = causal_contrast(apos, 0.0)
        @test nrow(ct) == 1
        @test abs(ct.coef[1] - 0.5) < 0.45
        @test ct.std_err[1] > 0
    end

    @testset "DID two-period ATT" begin
        data = make_did_data(n_obs=1000, dim_x=4, theta=-2.0; seed=511)
        did = DoubleMLDID(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=3, score="observational", trimming_threshold=0.05,
            rng=MersenneTwister(511),
        )
        fit!(did)
        @test isfinite(did.coef[1])
        @test abs(did.coef[1] - (-2.0)) < 0.8
        @test did.se[1] > 0
        # experimental score still runs
        did_e = DoubleMLDID(
            data, RidgeLearner(α=0.5), nothing;
            n_folds=3, score="experimental",
            rng=MersenneTwister(512),
        )
        fit!(did_e)
        @test isfinite(did_e.coef[1])
    end

    @testset "LPLR logistic PLR" begin
        data = make_lplr_data(n_obs=1200, dim_x=12, alpha=0.5; seed=601)
        lplr = DoubleMLLPLR(
            data,
            LogisticRegressionLearner(α=0.5),
            RidgeLearner(α=0.5),
            RidgeLearner(α=0.5);
            n_folds=3, score="instrument", rng=MersenneTwister(601),
        )
        fit!(lplr)
        @test isfinite(lplr.coef[1])
        @test lplr.se[1] > 0
        # order of magnitude of true alpha
        @test abs(lplr.coef[1] - 0.5) < 0.6
    end

    @testset "SSM sample selection MAR" begin
        data = make_ssm_data(n_obs=1500, dim_x=4, theta=1.0; seed=611)
        ssm = DoubleMLSSM(
            data,
            RidgeLearner(α=0.5),
            LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            n_folds=3, trimming_threshold=0.05, rng=MersenneTwister(611),
        )
        fit!(ssm)
        @test isfinite(ssm.coef[1])
        @test abs(ssm.coef[1] - 1.0) < 0.7
        @test ssm.se[1] > 0
    end

    @testset "SSM nonignorable nested CF" begin
        data = make_ssm_data(n_obs=1800, dim_x=4, theta=1.0; nonignorable=true, seed=612)
        @test n_instr(data) == 1
        ssm = DoubleMLSSM(
            data,
            RidgeLearner(α=0.5),
            LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            score="nonignorable", n_folds=3, trimming_threshold=0.05,
            rng=MersenneTwister(612),
        )
        fit!(ssm; store_models=true)
        @test isfinite(ssm.coef[1])
        @test abs(ssm.coef[1] - 1.0) < 1.0
        @test ssm.se[1] > 0
        @test haskey(ssm.models, "reps")
        @test !isempty(ssm.models["reps"])
    end

    @testset "DID multi Callaway–Sant'Anna" begin
        data = make_did_panel_data(n_id=280, n_t=4, dim_x=3, theta=2.0; seed=621)
        multi = DoubleMLDIDMulti(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=3, trimming_threshold=0.05,
            control_group="never_treated",
            gt_combinations=:standard,
            rng=MersenneTwister(621),
        )
        fit!(multi)
        @test multi.fitted
        @test length(multi.coef) >= 1
        @test all(isfinite, multi.coef)
        tab = att_table(multi)
        @test nrow(tab) == length(multi.coef)
        @test :event_time in propertynames(tab)
        # post ATT mean near theta
        post = tab[tab.post, :]
        @test abs(mean(post.coef) - 2.0) < 1.2

        # aggregations
        ag = aggregate(multi, :group)
        @test ag.method == "group"
        @test length(ag.coef) >= 1
        @test isfinite(ag.overall_coef)
        at = aggregate(multi, :time)
        @test at.method == "time"
        ae = aggregate(multi, :eventstudy)
        @test ae.method == "eventstudy"
        @test any(occursin("e=", n) for n in ae.names)
        st = summary_table(ag)
        @test any(st.name .== "overall")

        # not_yet_treated control
        multi2 = DoubleMLDIDMulti(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=3, control_group="not_yet_treated",
            gt_combinations=:universal,
            rng=MersenneTwister(622),
        )
        fit!(multi2)
        @test all(isfinite, multi2.coef)

        # anticipation
        multi3 = DoubleMLDIDMulti(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=3, anticipation_periods=0,
            gt_combinations=:all,
            rng=MersenneTwister(623),
        )
        fit!(multi3)
        @test length(multi3.gt_combos) >= length(multi.gt_combos)
    end

    @testset "RDD sharp" begin
        data = make_rdd_data(n_obs=2500, dim_x=3, tau=1.0, fuzzy=false; seed=631)
        rdd = DoubleMLRDD(
            data, RidgeLearner(α=0.5);
            cutoff=0.0, fuzzy=false, n_folds=3, n_iterations=2,
            fs_specification="cutoff", rng=MersenneTwister(631),
        )
        fit!(rdd)
        @test isfinite(rdd.coef[1])
        @test abs(rdd.coef[1] - 1.0) < 0.8
        @test rdd.se[1] > 0
        @test isfinite(rdd.h_used) && rdd.h_used > 0
        # RDFlex alias + fs_specification variants smoke
        rdd2 = RDFlex(
            data, RidgeLearner(α=0.5);
            cutoff=0.0, n_folds=3, n_iterations=1,
            fs_specification="cutoff and score", rng=MersenneTwister(632),
        )
        fit!(rdd2)
        @test isfinite(rdd2.coef[1])
        rdd3 = RDFlex(
            data, RidgeLearner(α=0.5);
            cutoff=0.0, n_folds=3, n_iterations=2,
            fs_specification="interacted cutoff and score", rng=MersenneTwister(633),
        )
        fit!(rdd3)
        @test isfinite(rdd3.coef[1])
    end

    @testset "DID multi unit IF, bootstrap, p_adjust" begin
        data = make_did_panel_data(n_id=200, n_t=4, dim_x=3, theta=2.0; seed=701)
        multi = DoubleMLDIDMulti(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=3, rng=MersenneTwister(701),
        )
        fit!(multi)
        @test size(multi.if_units, 1) == length(unique(data.id))
        @test size(multi.if_units, 3) == length(multi.coef)
        # joint IF aggregation SE finite and positive
        ag = aggregate(multi, :group)
        @test all(ag.se .> 0)
        @test ag.overall_se > 0
        # multiplier bootstrap + joint CI + p_adjust
        bootstrap!(multi; method="normal", n_rep_boot=100, rng=MersenneTwister(702))
        ci_j = confint(multi; joint=true)
        @test nrow(ci_j) == length(multi.coef)
        @test all(ci_j.joint)
        padj = p_adjust(multi; method=:holm)
        @test all(padj.pvalue_adjusted .>= padj.pvalue .- 1e-12)
        padj_rw = p_adjust(multi; method=:romano_wolf)
        @test all(0 .<= padj_rw.pvalue_adjusted .<= 1)
        padj_b = p_adjust(multi; method=:bonferroni)
        @test all(padj_b.pvalue_adjusted .>= padj_b.pvalue .- 1e-12)
    end

    @testset "DIDCS repeated cross-section" begin
        data = make_did_cs_data(n_obs=1200, dim_x=3, theta=-2.0; seed=711)
        dcs = DoubleMLDIDCS(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=3, score="observational", rng=MersenneTwister(711),
        )
        fit!(dcs)
        @test isfinite(dcs.coef[1])
        @test abs(dcs.coef[1] - (-2.0)) < 1.0
        @test dcs.se[1] > 0
        # experimental score
        dcs2 = DoubleMLDIDCS(
            data, RidgeLearner(α=0.5), nothing;
            n_folds=3, score="experimental", rng=MersenneTwister(712),
        )
        fit!(dcs2)
        @test isfinite(dcs2.coef[1])
    end

    @testset "Cluster SE" begin
        data = make_plr_data(n_obs=600, dim_x=5, theta=0.5; seed=721)
        dml = DoubleMLPLR(data, RidgeLearner(α=1.0), RidgeLearner(α=1.0);
                          n_folds=3, rng=MersenneTwister(721))
        fit!(dml)
        # synthetic clusters (block of 10)
        cluster = repeat(1:60, inner=10)
        r = cluster_se(dml; cluster=cluster)
        @test r.n_clusters == 60
        @test r.se[1] > 0
        @test isfinite(r.ci_lower[1])
        se_iid = dml.se[1]
        apply_cluster_se!(dml; cluster=cluster)
        @test dml.se[1] ≈ r.se[1]
        @test dml.se[1] > 0
        # cluster SE typically ≥ iid SE with positive within-cluster correlation
        @test dml.se[1] >= 0.5 * se_iid
    end

    @testset "PLPR approaches (fd / wg / cre)" begin
        data = make_plpr_data(n_id=180, n_t=4, dim_x=3, theta=0.5; seed=731)
        for ap in ("fd_exact", "wg_approx", "cre_general", "cre_normal")
            plpr = DoubleMLPLPR(
                data, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                approach=ap, n_folds=3, rng=MersenneTwister(731),
            )
            fit!(plpr)
            @test isfinite(plpr.coef[1])
            @test abs(plpr.coef[1] - 0.5) < 0.45
            @test plpr.se[1] > 0
            @test plpr.transformed !== nothing
            if ap in ("cre_general", "cre_normal")
                @test plpr.d_mean !== nothing
                @test length(plpr.d_mean) == length(plpr.transformed.y)
            end
        end
        # IV-type score on CRE
        plpr_iv = DoubleMLPLPR(
            data, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
            ml_g=RidgeLearner(α=0.5), approach="cre_general", score="IV-type",
            n_folds=3, rng=MersenneTwister(732),
        )
        fit!(plpr_iv)
        @test isfinite(plpr_iv.coef[1])
    end

    @testset "Base p_adjust and IRM weights" begin
        data = make_pliv_data(n_obs=800, dim_x=4, dim_z=1, theta=0.5; seed=741)
        # multi-coef via PLIV is single; use IRM + synthetic second model via p_adjust on multi DID already covered
        irm_data = make_irm_data(n_obs=800, dim_x=4, theta=0.5; seed=742)
        n = length(irm_data.y)
        w = ones(n)
        w[1:100] .= 1.5
        irm = DoubleMLIRM(
            irm_data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
            n_folds=3, weights=w, rng=MersenneTwister(742),
        )
        fit!(irm)
        @test isfinite(irm.coef[1])
        # single-parameter p_adjust
        pa = p_adjust(irm; method=:holm)
        @test pa.pvalue_adjusted[1] ≈ pa.pvalue[1]
        # set_sample_splitting!
        smpls = DoubleML.make_repeated_folds(n, 3, 1; rng=MersenneTwister(743))
        set_sample_splitting!(irm, smpls)
        @test irm.fitted == false
        fit!(irm)
        @test irm.fitted
    end

    @testset "Framework construct / algebra / concat" begin
        d1 = make_plr_data(n_obs=500, dim_x=5, theta=0.5; seed=801)
        d2 = make_plr_data(n_obs=500, dim_x=5, theta=1.0; seed=802)
        m1 = DoubleMLPLR(d1, RidgeLearner(α=1.0), RidgeLearner(α=1.0); n_folds=3, rng=MersenneTwister(801))
        m2 = DoubleMLPLR(d2, RidgeLearner(α=1.0), RidgeLearner(α=1.0); n_folds=3, rng=MersenneTwister(802))
        fit!(m1); fit!(m2)
        f1 = construct_framework(m1)
        f2 = construct_framework(m2)
        @test f1.thetas[1] ≈ m1.coef[1]
        @test f1.ses[1] ≈ m1.se[1]
        st = summary_table(f1)
        @test st.coef[1] ≈ m1.coef[1]

        fsum = f1 + f1
        @test fsum.thetas[1] ≈ 2 * f1.thetas[1]
        fdiff = f2 - f1
        @test isfinite(fdiff.thetas[1]) && fdiff.ses[1] > 0
        fsc = 2 * f1
        @test fsc.thetas[1] ≈ 2 * f1.thetas[1]
        @test fsc.ses[1] ≈ 2 * f1.ses[1]

        fc = concat([f1, f2])
        @test length(fc.thetas) == 2
        @test fc.thetas[1] ≈ f1.thetas[1]
        @test fc.thetas[2] ≈ f2.thetas[1]
        bootstrap!(fc; n_rep_boot=50, rng=MersenneTwister(803))
        cij = confint(fc; joint=true)
        @test nrow(cij) == 2
        padj = p_adjust(fc; method=:holm)
        @test nrow(padj) == 2
    end

    @testset "Multi-treatment PLR" begin
        θ_true = [0.5, -0.3]
        data = make_plr_multi_data(n_obs=1000, dim_x=6, theta=θ_true; seed=811)
        @test n_treat(data) == 2
        @test data.d_cols == ["d1", "d2"]
        plr = DoubleMLPLR(data, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                          n_folds=3, rng=MersenneTwister(811))
        fit!(plr)
        @test length(plr.coef) == 2
        @test abs(plr.coef[1] - θ_true[1]) < 0.2
        @test abs(plr.coef[2] - θ_true[2]) < 0.2
        @test all(plr.se .> 0)
        f = construct_framework(plr)
        @test length(f.thetas) == 2
        @test f.treatment_names == ["d1", "d2"]
    end

    @testset "Cluster-in-fit PLR" begin
        data = make_plr_cluster_data(n_obs=600, n_clusters=40, dim_x=4, theta=0.5; seed=821)
        @test is_cluster_data(data)
        plr = DoubleMLPLR(data, RidgeLearner(α=0.5), RidgeLearner(α=0.5);
                          n_folds=3, rng=MersenneTwister(821))
        fit!(plr)
        @test isfinite(plr.coef[1])
        @test abs(plr.coef[1] - 0.5) < 0.35
        @test plr.se[1] > 0
        @test plr.is_cluster_data
        @test plr.var_scaling !== nothing
        # cluster folds: no cluster id in both train and test
        folds = plr.smpls[1]
        cl = data.cluster[:, 1]
        for f in folds
            tr_c = Set(cl[f.train])
            te_c = Set(cl[f.test])
            @test isempty(intersect(tr_c, te_c))
        end
        f = construct_framework(plr)
        @test f.core.is_cluster_data
        bootstrap!(f; n_rep_boot=30, rng=MersenneTwister(822))
        @test confint(f; joint=false).lower[1] < plr.coef[1]
    end

    @testset "IIVM normalize_ipw / subgroups / evaluate_learners" begin
        data = make_iivm_data(n_obs=1800, dim_x=4, theta=0.5; seed=901)
        iivm = DoubleMLIIVM(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            n_folds=3, trimming_threshold=0.05,
            normalize_ipw=true,
            subgroups=(always_takers=true, never_takers=true),
            rng=MersenneTwister(901),
        )
        fit!(iivm)
        @test isfinite(iivm.coef[1])
        @test iivm.se[1] > 0
        ev = evaluate_learners(iivm)
        @test haskey(ev, "ml_m")
        @test ev["ml_m"] >= 0
        # no always-takers: r0 forced to 0
        iivm2 = DoubleMLIIVM(
            data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5),
            LogisticRegressionLearner(α=0.5);
            n_folds=3, always_takers=false, never_takers=true,
            rng=MersenneTwister(902),
        )
        fit!(iivm2)
        @test all(iivm2.predictions["ml_r0"] .== 0)
    end

    @testset "external_predictions PLR" begin
        data = make_plr_data(n_obs=400, dim_x=4, theta=0.5; seed=911)
        plr0 = DoubleMLPLR(data, LinearRegressionLearner(), LinearRegressionLearner();
                           n_folds=3, rng=MersenneTwister(911))
        fit!(plr0)
        # re-fit using stored predictions as external (should match closely)
        plr1 = DoubleMLPLR(data, LinearRegressionLearner(), LinearRegressionLearner();
                           n_folds=3, n_rep=1, draw_sample_splitting=false, rng=MersenneTwister(911))
        set_sample_splitting!(plr1, plr0.smpls)
        fit!(plr1; external_predictions=plr0.predictions)
        @test plr1.coef[1] ≈ plr0.coef[1] atol=1e-10
        @test plr1.se[1] ≈ plr0.se[1] atol=1e-10
    end

    @testset "DID multi experimental score" begin
        data = make_did_panel_data(n_id=180, n_t=4, dim_x=2, theta=2.0; seed=921)
        multi = DoubleMLDIDMulti(
            data, RidgeLearner(α=0.5), nothing;
            score="experimental", n_folds=3, rng=MersenneTwister(921),
        )
        fit!(multi)
        @test multi.fitted
        @test all(isfinite, multi.coef)
        post = att_table(multi)
        @test abs(mean(post.coef[post.post]) - 2.0) < 1.5
        et = effects_table(multi; method=:eventstudy)
        @test nrow(et) >= 1
        @test :ci_lower in propertynames(et)
        @test plot_effects(multi; method=:group) isa DataFrame
    end

    @testset "set_ml_nuisance_params PLR" begin
        data = make_plr_data(n_obs=400, dim_x=4, theta=0.5; seed=931)
        plr = DoubleMLPLR(data, RidgeLearner(α=1.0), RidgeLearner(α=1.0);
                          n_folds=3, rng=MersenneTwister(931))
        set_ml_nuisance_params!(plr, "ml_l", data.d_col, Dict(:α => 0.1))
        set_ml_nuisance_params!(plr, "ml_m", data.d_col, Dict(:α => 0.1))
        fit!(plr)
        @test isfinite(plr.coef[1])
        @test haskey(plr.ml_params, "ml_l")
    end

    @testset "IRM store_models + sensitivity_contour + tune extensions" begin
        data = make_irm_data(n_obs=600, dim_x=4, theta=0.5; seed=941)
        irm = DoubleMLIRM(data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
                          n_folds=3, rng=MersenneTwister(941))
        set_ml_nuisance_params!(irm, "ml_m", data.d_col, Dict(:α => 0.2))
        fit!(irm; store_models=true)
        @test haskey(irm.models, "ml_m")
        @test irm.models["ml_m"] !== nothing
        sensitivity_analysis!(irm; cf_y=0.04, cf_d=0.03)
        grid = sensitivity_contour(irm; cf_y_max=0.1, cf_d_max=0.1, grid_size=5)
        @test nrow(grid) == 25
        @test :covers_null in propertynames(grid)
        @test all(isfinite, grid.theta_lower)

        # tune SSM / DID / PLPR smoke
        sdata = make_ssm_data(n_obs=500, dim_x=3, theta=1.0; seed=942)
        ssm = DoubleMLSSM(sdata, RidgeLearner(α=1.0), LogisticRegressionLearner(α=1.0),
                          LogisticRegressionLearner(α=1.0); n_folds=3, rng=MersenneTwister(942))
        tr = tune!(ssm; param_grids=Dict(:ml_m => Dict(:α => [0.1, 1.0])), n_folds_tune=3)
        @test haskey(tr, :ml_m)
        fit!(ssm)
        @test isfinite(ssm.coef[1])

        ddata = make_did_data(n_obs=400, theta=-2.0; seed=943)
        did = DoubleMLDID(ddata, RidgeLearner(α=1.0), LogisticRegressionLearner(α=1.0);
                          n_folds=3, rng=MersenneTwister(943))
        tr2 = tune!(did; param_grids=Dict(:ml_g => Dict(:α => [0.1, 1.0])), n_folds_tune=3)
        @test haskey(tr2, :ml_g)

        pdata = make_plpr_data(n_id=80, n_t=3, dim_x=2, theta=0.5; seed=944)
        plpr = DoubleMLPLPR(pdata, RidgeLearner(α=1.0), RidgeLearner(α=1.0);
                            approach="wg_approx", n_folds=3, rng=MersenneTwister(944))
        tr3 = tune!(plpr; param_grids=Dict(:ml_l => Dict(:α => [0.1, 1.0])), n_folds_tune=3)
        @test haskey(tr3, :ml_l)
        fit!(plpr)
        @test isfinite(plpr.coef[1])

        # PLIV store_models
        iv = make_pliv_data(n_obs=500, dim_x=4, dim_z=1, theta=0.5; seed=945)
        ml = RidgeLearner(α=0.5)
        pliv = DoubleMLPLIV(iv, clone(ml), clone(ml), clone(ml); n_folds=3, rng=MersenneTwister(945))
        set_ml_nuisance_params!(pliv, "ml_l", iv.d_col, Dict(:α => 0.2))
        fit!(pliv; store_models=true)
        @test haskey(pliv.models, "ml_l")
        @test length(pliv.models["ml_l"]) == 3
    end

    @testset "PSProcessor clips propensities" begin
        p = PSProcessor(clipping_threshold=0.05, extreme_threshold=1e-8)
        @test p isa PSProcessorConfig  # alias
        raw = [0.0, 0.5, 1.0, -0.1, 1.2]
        out = process_propensity(raw, p)
        @test all(out .>= 0.05 - 1e-12)
        @test all(out .<= 0.95 + 1e-12)
        out2 = process_propensity(raw, 0.01)
        @test minimum(out2) ≈ 0.01 atol=1e-12

        data = make_irm_data(n_obs=500, dim_x=3, theta=0.5; seed=951)
        irm = DoubleMLIRM(data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
                          n_folds=3, ps_processor=PSProcessor(clipping_threshold=0.02),
                          rng=MersenneTwister(951))
        fit!(irm)
        @test isfinite(irm.coef[1])
        @test irm.ps_processor.clipping_threshold == 0.02
    end

    @testset "Framework sensitivity from PLR" begin
        data = make_plr_data(n_obs=600, dim_x=5, theta=0.5; seed=952)
        plr = DoubleMLPLR(data, LinearRegressionLearner(), LinearRegressionLearner();
                          n_folds=3, rng=MersenneTwister(952))
        fit!(plr)
        sensitivity_analysis!(plr; cf_y=0.04, cf_d=0.03)
        fw = construct_framework(plr)
        @test fw.sens_elements !== nothing
        r = sensitivity_analysis!(fw; cf_y=0.04, cf_d=0.03)
        @test r.theta_lower[1] ≈ plr.sensitivity.theta_lower[1] atol=1e-10
        @test r.rv[1] ≈ plr.sensitivity.rv[1] atol=1e-6
        grid = sensitivity_plot(fw; cf_y_max=0.1, cf_d_max=0.1, grid_size=4)
        @test nrow(grid) == 16
        s = sensitivity_summary(fw)
        @test occursin("Robustness", s)
        # arithmetic drops sens
        fw2 = 2 * fw
        @test fw2.sens_elements === nothing
    end

    @testset "Confounded / heterogeneous / discrete DGPs" begin
        cplr = make_confounded_plr_data(n_obs=400, theta=2.0, cf_y=0.05, cf_d=0.05; seed=953)
        @test length(cplr.y) == 400
        @test cplr.data isa DoubleMLData
        plr = DoubleMLPLR(cplr.data, LinearRegressionLearner(), LinearRegressionLearner();
                          n_folds=3, rng=MersenneTwister(953))
        fit!(plr)
        @test isfinite(plr.coef[1])
        sensitivity_analysis!(plr; cf_y=0.05, cf_d=0.05)
        @test plr.sensitivity.rv[1] > 0

        cirm = make_confounded_irm_data(n_obs=400, theta=0.5, linear=true; seed=954)
        irm = DoubleMLIRM(cirm.data, RidgeLearner(α=0.5), LogisticRegressionLearner(α=0.5);
                          n_folds=3, rng=MersenneTwister(954))
        fit!(irm)
        @test isfinite(irm.coef[1])

        het = make_heterogeneous_data(n_obs=300, p=10, support_size=3, n_x=1,
                                      binary_treatment=true; seed=955)
        @test length(het.effects) == 300
        @test het.dml_data isa DoubleMLData
        @test Set(unique(het.dml_data.d)) ⊆ Set([0.0, 1.0])

        disc = make_irm_data_discrete_treatments(n_obs=250, n_levels=3; seed=956)
        @test length(unique(disc.d)) >= 2
        @test length(disc.d_cont) == 250
    end
end
