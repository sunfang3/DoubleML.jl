# ---------------------------------------------------------------------------
# Nuisance learner protocol (sklearn-style duck typing)
#
# Required methods for any learner used as a DoubleML nuisance model:
#   fit!(learner, X, y)           — train in-place; return learner
#   predict(learner, X)           — continuous predictions (regressors)
#   clone(learner)                — unfitted deep-ish copy
#
# Classifiers additionally implement:
#   predict_proba(learner, X)     — P(y = 1) as a Vector
#   is_classifier(::Type) / is_classifier(learner) :: Bool
# ---------------------------------------------------------------------------

abstract type AbstractLearner end

"""Return `true` if the learner is a classifier (uses `predict_proba`)."""
is_classifier(::AbstractLearner) = false
is_classifier(x) = false

"""Deep-ish copy of an unfitted learner (reset fitted state)."""
function clone end

# ========================= Linear / Ridge regression =========================

"""
    RidgeLearner(; α=1.0, fit_intercept=true)

Closed-form ridge regression:
`β = (X'X + α I) \\ X'y` (with optional intercept column).
"""
mutable struct RidgeLearner <: AbstractLearner
    α::Float64
    fit_intercept::Bool
    coef::Vector{Float64}
    intercept::Float64
    fitted::Bool
end

function RidgeLearner(; α::Real=1.0, fit_intercept::Bool=true)
    RidgeLearner(Float64(α), fit_intercept, Float64[], 0.0, false)
end

LinearRegressionLearner(; fit_intercept::Bool=true) = RidgeLearner(; α=0.0, fit_intercept)

function clone(m::RidgeLearner)
    RidgeLearner(m.α, m.fit_intercept, Float64[], 0.0, false)
end

function fit!(m::RidgeLearner, X::AbstractMatrix, y::AbstractVector)
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X and y size mismatch"))
    yf = Float64.(y)
    if m.fit_intercept
        Xd = hcat(ones(n), Float64.(X))
        λ = m.α
        # do not penalize intercept
        pen = Diagonal(vcat(0.0, fill(λ, p)))
        β = (Xd' * Xd + pen) \ (Xd' * yf)
        m.intercept = β[1]
        m.coef = β[2:end]
    else
        pen = m.α * I(p)
        m.coef = (Float64.(X)' * Float64.(X) + pen) \ (Float64.(X)' * yf)
        m.intercept = 0.0
    end
    m.fitted = true
    return m
end

function predict(m::RidgeLearner, X::AbstractMatrix)
    m.fitted || error("RidgeLearner is not fitted")
    return m.intercept .+ Float64.(X) * m.coef
end

# ========================= Logistic regression (L2) =========================

"""
    LogisticRegressionLearner(; α=1.0, fit_intercept=true, max_iter=100, tol=1e-6)

L2-regularized logistic regression via IRLS / Newton steps.
Outputs `predict_proba` = P(y=1).
"""
mutable struct LogisticRegressionLearner <: AbstractLearner
    α::Float64
    fit_intercept::Bool
    max_iter::Int
    tol::Float64
    coef::Vector{Float64}
    intercept::Float64
    fitted::Bool
end

function LogisticRegressionLearner(; α::Real=1.0, fit_intercept::Bool=true,
                                   max_iter::Int=100, tol::Real=1e-6)
    LogisticRegressionLearner(Float64(α), fit_intercept, max_iter, Float64(tol),
                              Float64[], 0.0, false)
end

is_classifier(::LogisticRegressionLearner) = true

function clone(m::LogisticRegressionLearner)
    LogisticRegressionLearner(m.α, m.fit_intercept, m.max_iter, m.tol, Float64[], 0.0, false)
end

function _sigmoid(z)
    # numerically stable
    if z >= 0
        ez = exp(-z)
        return 1 / (1 + ez)
    else
        ez = exp(z)
        return ez / (1 + ez)
    end
end

function fit!(m::LogisticRegressionLearner, X::AbstractMatrix, y::AbstractVector)
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X and y size mismatch"))
    Xf = Float64.(X)
    yf = Float64.(y)
    # map labels to {0,1} if needed
    um = unique(yf)
    if Set(um) == Set([-1.0, 1.0])
        yf = (yf .+ 1) ./ 2
    end

    d = m.fit_intercept ? p + 1 : p
    β = zeros(d)
    Xd = m.fit_intercept ? hcat(ones(n), Xf) : Xf

    for _ in 1:m.max_iter
        η = Xd * β
        μ = _sigmoid.(η)
        w = μ .* (1 .- μ)
        w = clamp.(w, 1e-6, Inf)  # avoid singular W
        z = η .+ (yf .- μ) ./ w
        # weighted ridge: (X' W X + α I_{-int}) β = X' W z
        WX = Xd .* sqrt.(w)
        Wy = z .* sqrt.(w)
        A = WX' * WX
        if m.fit_intercept
            for j in 2:d
                A[j, j] += m.α
            end
        else
            for j in 1:d
                A[j, j] += m.α
            end
        end
        β_new = A \ (WX' * Wy)
        if norm(β_new - β) < m.tol
            β = β_new
            break
        end
        β = β_new
    end

    if m.fit_intercept
        m.intercept = β[1]
        m.coef = β[2:end]
    else
        m.intercept = 0.0
        m.coef = β
    end
    m.fitted = true
    return m
end

function predict_proba(m::LogisticRegressionLearner, X::AbstractMatrix)
    m.fitted || error("LogisticRegressionLearner is not fitted")
    η = m.intercept .+ Float64.(X) * m.coef
    return _sigmoid.(η)
end

function predict(m::LogisticRegressionLearner, X::AbstractMatrix)
    return Float64.(predict_proba(m, X) .>= 0.5)
end

# ========================= Random forests (DecisionTree.jl) =========================

"""
    RandomForestRegressorLearner(; n_trees=100, max_depth=-1, min_samples_leaf=1, rng=Random.default_rng())

Wrapper around `DecisionTree.RandomForestRegressor` (pure Julia, actively maintained).
"""
mutable struct RandomForestRegressorLearner <: AbstractLearner
    n_trees::Int
    max_depth::Int
    min_samples_leaf::Int
    rng::AbstractRNG
    model::Any
    fitted::Bool
end

function RandomForestRegressorLearner(; n_trees::Int=100, max_depth::Int=-1,
                                      min_samples_leaf::Int=1,
                                      rng::AbstractRNG=Random.default_rng())
    RandomForestRegressorLearner(n_trees, max_depth, min_samples_leaf, rng, nothing, false)
end

function clone(m::RandomForestRegressorLearner)
    RandomForestRegressorLearner(m.n_trees, m.max_depth, m.min_samples_leaf,
                                 copy(m.rng), nothing, false)
end

function fit!(m::RandomForestRegressorLearner, X::AbstractMatrix, y::AbstractVector)
    # DecisionTree expects Matrix{Float64} and labels as Vector
    Xf = Matrix{Float64}(X)
    yf = Float64.(y)
    m.model = build_forest(yf, Xf, -1, m.n_trees, 0.7, m.max_depth;
                           rng=m.rng)
    # build_forest for regression: when labels are Float64 it does regression forest
    m.fitted = true
    return m
end

function predict(m::RandomForestRegressorLearner, X::AbstractMatrix)
    m.fitted || error("RandomForestRegressorLearner is not fitted")
    Xf = Matrix{Float64}(X)
    return apply_forest(m.model, Xf)
end

"""
    RandomForestClassifierLearner(; n_trees=100, max_depth=-1, min_samples_leaf=1, rng=...)

Random forest classifier; `predict_proba` returns P(class = 1) for binary labels in {0,1}.
"""
mutable struct RandomForestClassifierLearner <: AbstractLearner
    n_trees::Int
    max_depth::Int
    min_samples_leaf::Int
    rng::AbstractRNG
    model::Any
    classes::Vector
    fitted::Bool
end

function RandomForestClassifierLearner(; n_trees::Int=100, max_depth::Int=-1,
                                       min_samples_leaf::Int=1,
                                       rng::AbstractRNG=Random.default_rng())
    RandomForestClassifierLearner(n_trees, max_depth, min_samples_leaf, rng,
                                  nothing, [], false)
end

is_classifier(::RandomForestClassifierLearner) = true

function clone(m::RandomForestClassifierLearner)
    RandomForestClassifierLearner(m.n_trees, m.max_depth, m.min_samples_leaf,
                                  copy(m.rng), nothing, [], false)
end

function fit!(m::RandomForestClassifierLearner, X::AbstractMatrix, y::AbstractVector)
    Xf = Matrix{Float64}(X)
    # DecisionTree classification wants categorical labels
    y_int = Int.(round.(y))
    m.classes = sort(unique(y_int))
    n_subfeatures = -1
    m.model = build_forest(y_int, Xf, n_subfeatures, m.n_trees, 0.7, m.max_depth;
                           rng=m.rng)
    m.fitted = true
    return m
end

function predict(m::RandomForestClassifierLearner, X::AbstractMatrix)
    m.fitted || error("RandomForestClassifierLearner is not fitted")
    Xf = Matrix{Float64}(X)
    return Float64.(apply_forest(m.model, Xf))
end

function predict_proba(m::RandomForestClassifierLearner, X::AbstractMatrix)
    m.fitted || error("RandomForestClassifierLearner is not fitted")
    Xf = Matrix{Float64}(X)
    # apply_forest_proba returns n × n_classes
    # DecisionTree.jl: apply_forest_proba(forest, features, list of labels)
    proba = apply_forest_proba(m.model, Xf, m.classes)
    # find column for class 1
    idx = findfirst(==(1), m.classes)
    if idx === nothing
        # no positive class seen in training — return zeros
        return zeros(size(X, 1))
    end
    return vec(proba[:, idx])
end

# ========================= Cross-fit helper =========================

"""
Out-of-fold predictions for a learner using precomputed folds.

`folds` is a vector of `(train_idx, test_idx)` pairs covering all observations.
"""
function cross_fit_predict(learner, X::AbstractMatrix, y::AbstractVector,
                           folds; classifier::Bool=false)
    n = size(X, 1)
    preds = fill(NaN, n)
    for (train, test) in folds
        m = clone(learner)
        fit!(m, X[train, :], y[train])
        if classifier || is_classifier(m)
            preds[test] = predict_proba(m, X[test, :])
        else
            preds[test] = predict(m, X[test, :])
        end
    end
    return preds
end
