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
end
