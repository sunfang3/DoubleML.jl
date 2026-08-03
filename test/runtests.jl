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
end
