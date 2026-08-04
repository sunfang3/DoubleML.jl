using Test
using Random
using Statistics
using DoubleML
using MLJBase
using MLJLinearModels

@testset "MLJ learner adapter" begin
    X = randn(MersenneTwister(1201), 160, 4)
    y = X[:, 1] .- 0.5 .* X[:, 2] .+ randn(MersenneTwister(1202), 160)
    d = Float64.(X[:, 3] .+ randn(MersenneTwister(1203), 160) .> 0)
    data = DoubleMLData(X, y, d)

    ml_y = MLJLearner(LinearRegressor())
    ml_d = MLJLearner(LogisticClassifier(); classifier=true, positive_label=1.0)
    model = DoubleMLPLR(data, clone(ml_y), clone(ml_d); n_folds=4,
                        rng=MersenneTwister(1204))
    DoubleML.fit!(model)
    @test model.fitted
    @test isfinite(model.coef[1])
    @test model.se[1] > 0

    task = NuisanceTask(:outcome, clone(ml_y), data.y)
    plan = CrossFitPlan(data, 4, 1; rng=MersenneTwister(1205))
    out = fit_nuisance(plan, task, data.x)
    @test size(out.predictions) == (160, 1)
    @test all(isfinite, out.predictions)
end
