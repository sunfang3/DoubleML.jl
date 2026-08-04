"""Shared cross-fitting primitives used by estimator implementations."""

"""
    CrossFitPlan

Validated sample splits for one DML estimation run. The causal package owns
these splits; learner backends only receive each training/test partition.
"""
struct CrossFitPlan
    smpls::Vector
    smpls_cluster::Union{Nothing,Vector}
    n_obs::Int
    n_folds::Int
    n_rep::Int
    n_folds_per_cluster::Int
end

function CrossFitPlan(data::DoubleMLData, n_folds::Int, n_rep::Int;
                      rng::AbstractRNG=Random.default_rng())
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))
    n_rep >= 1 || throw(ArgumentError("n_rep must be ≥ 1"))
    smpls, smpls_cluster, n_fpc = init_sample_splitting(data, n_folds, n_rep; rng=rng)
    return CrossFitPlan(smpls, smpls_cluster, n_obs(data), n_folds, n_rep, n_fpc)
end

Base.length(plan::CrossFitPlan) = plan.n_rep
folds(plan::CrossFitPlan, rep::Int) = plan.smpls[rep]

"""Description of one nuisance regression/classification task."""
struct NuisanceTask
    name::Symbol
    learner::Any
    target::AbstractVector
    classifier::Bool
    train_rows::Function
    predict_transform::Function
end

function NuisanceTask(name, learner, target::AbstractVector;
                      classifier::Bool=false,
                      train_rows::Function=identity,
                      predict_transform::Function=identity)
    return NuisanceTask(Symbol(name), learner, target, classifier,
                        train_rows, predict_transform)
end

"""Cross-fitted predictions and optional fitted fold models."""
struct CrossFitResult
    predictions::Matrix{Float64}
    models::Union{Nothing,Vector{Any}}
end

function _fit_nuisance_task(task::NuisanceTask, X::AbstractMatrix, folds;
                            params_factory=nothing, store_models::Bool=false)
    n = size(X, 1)
    preds = fill(NaN, n)
    models = store_models ? Any[] : nothing
    for (k, fold) in enumerate(folds)
        train, test = hasproperty(fold, :train) ?
            (fold.train, fold.test) : (fold[1], fold[2])
        rows = task.train_rows(train)
        p = params_factory === nothing ? nothing : params_factory(k)
        model = p === nothing ? clone(task.learner) : _clone_with_params(task.learner, p)
        fit!(model, X[rows, :], task.target[rows])
        raw = task.classifier || is_classifier(model) ?
            predict_proba(model, X[test, :]) : predict(model, X[test, :])
        pred = Float64.(task.predict_transform(raw))
        length(pred) == length(test) || throw(DimensionMismatch(
            "nuisance task $(task.name) returned $(length(pred)) predictions for " *
            "$(length(test)) test rows"))
        preds[test] = pred
        store_models && push!(models, model)
    end
    all(isfinite, preds) || throw(ArgumentError(
        "nuisance task $(task.name) did not produce a prediction for every observation"))
    return preds, models
end

"""Run one nuisance task over all repetitions in a cross-fitting plan."""
function fit_nuisance(plan::CrossFitPlan, task::NuisanceTask, X::AbstractMatrix;
                      params_factory=nothing, store_models::Bool=false)
    size(X, 1) == plan.n_obs || throw(DimensionMismatch("X rows must match CrossFitPlan"))
    size(task.target, 1) == plan.n_obs || throw(DimensionMismatch(
        "nuisance target length must match CrossFitPlan"))
    preds = Matrix{Float64}(undef, plan.n_obs, plan.n_rep)
    models = store_models ? Vector{Any}(undef, plan.n_rep) : nothing
    for r in 1:plan.n_rep
        factory = params_factory === nothing ? nothing :
            (k -> params_factory(r, k))
        preds[:, r], fold_models = _fit_nuisance_task(
            task, X, plan.smpls[r]; params_factory=factory, store_models=store_models)
        store_models && (models[r] = fold_models)
    end
    return CrossFitResult(preds, models)
end

"""Internal one-repetition adapter used by the legacy public helpers."""
function _cross_fit_one(learner, X::AbstractMatrix, y::AbstractVector, folds;
                        classifier::Bool=false, params_factory=nothing,
                        store_models::Bool=false)
    task = NuisanceTask(:anonymous, learner, y; classifier=classifier)
    return _fit_nuisance_task(task, X, folds;
                              params_factory=params_factory,
                              store_models=store_models)
end
