# Hyperparameter tuning for nuisance learners and DoubleML models
# (Python DoubleML.tune / grid_search analogue — lightweight, no external deps)

"""
    TuneResult

Result of tuning a single learner.
"""
struct TuneResult
    best_params::Dict{Symbol,Any}
    best_score::Float64          # lower is better (RMSE or log-loss)
    all_params::Vector{Dict{Symbol,Any}}
    all_scores::Vector{Float64}
end

function Base.show(io::IO, r::TuneResult)
    print(io, "TuneResult(best_score=$(round(r.best_score; digits=5)), best_params=$(r.best_params))")
end

"""Cartesian product of named parameter grids."""
function expand_param_grid(grid::AbstractDict{Symbol,<:AbstractVector})
    isempty(grid) && return [Dict{Symbol,Any}()]
    keys_ = collect(keys(grid))
    vals = [collect(grid[k]) for k in keys_]
    combos = Vector{Dict{Symbol,Any}}()
    # recursive product
    function rec(i, cur)
        if i > length(keys_)
            push!(combos, copy(cur))
            return
        end
        for v in vals[i]
            cur[keys_[i]] = v
            rec(i + 1, cur)
        end
    end
    rec(1, Dict{Symbol,Any}())
    return combos
end

"""
Cross-validated prediction score for a learner (lower = better).

- Regressors: RMSE
- Classifiers: binary log-loss
"""
function cv_score(learner, X::AbstractMatrix, y::AbstractVector, folds;
                  classifier::Bool=is_classifier(learner))
    preds = cross_fit_predict(learner, X, y, folds; classifier=classifier)
    if classifier
        p = clamp.(preds, 1e-15, 1 - 1e-15)
        yf = Float64.(y)
        return -mean(yf .* log.(p) .+ (1 .- yf) .* log.(1 .- p))
    else
        return sqrt(mean((Float64.(y) .- preds) .^ 2))
    end
end

"""
    tune_learner(learner, X, y, param_grid; n_folds=5, search_mode=:grid,
                 n_iter=10, rng=...)

Tune hyperparameters of a single nuisance learner.

# Arguments
- `param_grid::Dict{Symbol,Vector}` — e.g. `Dict(:α => [0.1, 1.0, 10.0])`
- `search_mode` — `:grid` (full product) or `:random` (sample `n_iter` combos)
- Returns `(best_learner, TuneResult)` where `best_learner` is a clone with
  best params applied (not yet fitted on full data).
"""
function tune_learner(learner, X::AbstractMatrix, y::AbstractVector,
                      param_grid::AbstractDict{Symbol,<:AbstractVector};
                      n_folds::Int=5,
                      search_mode::Symbol=:grid,
                      n_iter::Int=10,
                      rng::AbstractRNG=Random.default_rng(),
                      classifier::Bool=is_classifier(learner))
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))
    combos = expand_param_grid(param_grid)
    if search_mode === :random
        n_iter = min(n_iter, length(combos))
        combos = combos[randperm(rng, length(combos))[1:n_iter]]
    elseif search_mode !== :grid
        throw(ArgumentError("search_mode must be :grid or :random"))
    end

    folds = make_folds(size(X, 1), n_folds; rng=rng)
    scores = Float64[]
    best_score = Inf
    best_params = Dict{Symbol,Any}()
    best_learner = clone(learner)

    for params in combos
        cand = clone(learner)
        set_params!(cand; params...)
        sc = cv_score(cand, X, y, folds; classifier=classifier)
        push!(scores, sc)
        if sc < best_score
            best_score = sc
            best_params = params
            best_learner = cand
        end
    end

    result = TuneResult(best_params, best_score, combos, scores)
    return best_learner, result
end

"""
    tune!(m::DoubleMLPLR; param_grids, n_folds_tune=5, search_mode=:grid, n_iter=10)

Tune nuisance learners of a PLR model (Python `DoubleMLPLR.tune` analogue).

`param_grids` keys: `:ml_l`, `:ml_m`, and optionally `:ml_g` (IV-type).

After tuning, the model's learners are replaced by the best-parameter clones.
Call `fit!` afterwards to estimate the causal parameter.

# Example
```julia
dml = DoubleMLPLR(data, RidgeLearner(α=1.0), RidgeLearner(α=1.0))
tune!(dml; param_grids=Dict(
    :ml_l => Dict(:α => [0.01, 0.1, 1.0, 10.0]),
    :ml_m => Dict(:α => [0.01, 0.1, 1.0, 10.0]),
))
fit!(dml)
```
"""
function tune!(m::DoubleMLPLR;
               param_grids::AbstractDict,
               n_folds_tune::Int=5,
               search_mode::Symbol=:grid,
               n_iter::Int=10,
               rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    results = Dict{Symbol,TuneResult}()

    if haskey(param_grids, :ml_l)
        best, res = tune_learner(m.ml_l, X, y, param_grids[:ml_l];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=false)
        m.ml_l = best
        results[:ml_l] = res
    end
    if haskey(param_grids, :ml_m)
        best, res = tune_learner(m.ml_m, X, d, param_grids[:ml_m];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng,
                                 classifier=is_classifier(m.ml_m))
        m.ml_m = best
        results[:ml_m] = res
    end
    if haskey(param_grids, :ml_g) && m.ml_g !== nothing
        # rough residual target for IV-type
        best_l = haskey(results, :ml_l) ? m.ml_l : m.ml_l
        best_m = haskey(results, :ml_m) ? m.ml_m : m.ml_m
        # use a simple residual if we can fit quickly
        folds = make_folds(n_obs(data), n_folds_tune; rng=rng)
        ℓ̂ = cross_fit_predict(best_l, X, y, folds)
        m̂ = cross_fit_predict(best_m, X, d, folds; classifier=is_classifier(best_m))
        v = d .- m̂; u = y .- ℓ̂
        θ0 = sum(v .* u) / max(sum(v .* v), eps())
        best, res = tune_learner(m.ml_g, X, y .- θ0 .* d, param_grids[:ml_g];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=false)
        m.ml_g = best
        results[:ml_g] = res
    end

    m.fitted = false
    m.boot = nothing
    return results
end

"""
    tune!(m::DoubleMLIRM; param_grids, ...)

Tune IRM nuisances. Keys: `:ml_g`, `:ml_m`.
"""
function tune!(m::DoubleMLIRM;
               param_grids::AbstractDict,
               n_folds_tune::Int=5,
               search_mode::Symbol=:grid,
               n_iter::Int=10,
               rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    results = Dict{Symbol,TuneResult}()

    if haskey(param_grids, :ml_g)
        best, res = tune_learner(m.ml_g, X, y, param_grids[:ml_g];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=false)
        m.ml_g = best
        results[:ml_g] = res
    end
    if haskey(param_grids, :ml_m)
        best, res = tune_learner(m.ml_m, X, d, param_grids[:ml_m];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng,
                                 classifier=is_classifier(m.ml_m))
        m.ml_m = best
        results[:ml_m] = res
    end

    m.fitted = false
    m.boot = nothing
    return results
end

"""
    tune!(m::DoubleMLPLIV; param_grids, ...)

Tune PLIV nuisances. Keys depend on `partial_mode`:
- `:partialX` — `:ml_l`, `:ml_m`, `:ml_r` (and `:ml_g` for IV-type)
- `:partialZ` — `:ml_r`
- `:partialXZ` — `:ml_l`, `:ml_m`, `:ml_r`
"""
function tune!(m::DoubleMLPLIV;
               param_grids::AbstractDict,
               n_folds_tune::Int=5,
               search_mode::Symbol=:grid,
               n_iter::Int=10,
               rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    Z = data.z
    results = Dict{Symbol,TuneResult}()

    if m.partial_mode == :partialZ
        XZ = hcat(X, Z)
        if haskey(param_grids, :ml_r)
            best, res = tune_learner(m.ml_r, XZ, d, param_grids[:ml_r];
                                     n_folds=n_folds_tune, search_mode=search_mode,
                                     n_iter=n_iter, rng=rng, classifier=false)
            m.ml_r = best
            results[:ml_r] = res
        end
    else
        if haskey(param_grids, :ml_l) && m.ml_l !== nothing
            best, res = tune_learner(m.ml_l, X, y, param_grids[:ml_l];
                                     n_folds=n_folds_tune, search_mode=search_mode,
                                     n_iter=n_iter, rng=rng, classifier=false)
            m.ml_l = best
            results[:ml_l] = res
        end
        if m.partial_mode == :partialX
            if haskey(param_grids, :ml_r) && m.ml_r !== nothing
                best, res = tune_learner(m.ml_r, X, d, param_grids[:ml_r];
                                         n_folds=n_folds_tune, search_mode=search_mode,
                                         n_iter=n_iter, rng=rng, classifier=false)
                m.ml_r = best
                results[:ml_r] = res
            end
            if haskey(param_grids, :ml_m) && m.ml_m !== nothing
                z = n_instr(data) == 1 ? vec(Z) : Z[:, 1]
                best, res = tune_learner(m.ml_m, X, z, param_grids[:ml_m];
                                         n_folds=n_folds_tune, search_mode=search_mode,
                                         n_iter=n_iter, rng=rng, classifier=false)
                m.ml_m = best
                results[:ml_m] = res
            end
            if haskey(param_grids, :ml_g) && m.ml_g !== nothing
                best, res = tune_learner(m.ml_g, X, y, param_grids[:ml_g];
                                         n_folds=n_folds_tune, search_mode=search_mode,
                                         n_iter=n_iter, rng=rng, classifier=false)
                m.ml_g = best
                results[:ml_g] = res
            end
        elseif m.partial_mode == :partialXZ
            XZ = hcat(X, Z)
            if haskey(param_grids, :ml_m) && m.ml_m !== nothing
                best, res = tune_learner(m.ml_m, XZ, d, param_grids[:ml_m];
                                         n_folds=n_folds_tune, search_mode=search_mode,
                                         n_iter=n_iter, rng=rng, classifier=false)
                m.ml_m = best
                results[:ml_m] = res
            end
            if haskey(param_grids, :ml_r) && m.ml_r !== nothing
                folds = make_folds(n_obs(data), n_folds_tune; rng=rng)
                mhat = cross_fit_predict(m.ml_m, XZ, d, folds)
                best, res = tune_learner(m.ml_r, X, mhat, param_grids[:ml_r];
                                         n_folds=n_folds_tune, search_mode=search_mode,
                                         n_iter=n_iter, rng=rng, classifier=false)
                m.ml_r = best
                results[:ml_r] = res
            end
        end
    end

    m.fitted = false
    m.boot = nothing
    return results
end

"""
    tune!(m::DoubleMLIIVM; param_grids, ...)

Tune IIVM nuisances. Keys: `:ml_g`, `:ml_m`, `:ml_r`.
"""
function tune!(m::DoubleMLIIVM;
               param_grids::AbstractDict,
               n_folds_tune::Int=5,
               search_mode::Symbol=:grid,
               n_iter::Int=10,
               rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    z = vec(data.z)
    results = Dict{Symbol,TuneResult}()

    if haskey(param_grids, :ml_g)
        best, res = tune_learner(m.ml_g, X, y, param_grids[:ml_g];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=false)
        m.ml_g = best
        results[:ml_g] = res
    end
    if haskey(param_grids, :ml_m)
        best, res = tune_learner(m.ml_m, X, z, param_grids[:ml_m];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng,
                                 classifier=is_classifier(m.ml_m))
        m.ml_m = best
        results[:ml_m] = res
    end
    if haskey(param_grids, :ml_r)
        best, res = tune_learner(m.ml_r, X, d, param_grids[:ml_r];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng,
                                 classifier=is_classifier(m.ml_r))
        m.ml_r = best
        results[:ml_r] = res
    end

    m.fitted = false
    m.boot = nothing
    return results
end

"""
    tune!(m::DoubleMLSSM; param_grids, ...)

Tune SSM nuisances. Keys: `:ml_g`, `:ml_m`, `:ml_pi`.
"""
function tune!(m::DoubleMLSSM;
               param_grids::AbstractDict,
               n_folds_tune::Int=5,
               search_mode::Symbol=:grid,
               n_iter::Int=10,
               rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d, s = data.x, data.y, data.d, data.s
    results = Dict{Symbol,TuneResult}()
    sel = s .== 1
    if haskey(param_grids, :ml_g) && any(sel)
        nf = min(n_folds_tune, max(2, count(sel) ÷ 10))
        best, res = tune_learner(m.ml_g, X[sel, :], y[sel], param_grids[:ml_g];
                                 n_folds=nf, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=false)
        m.ml_g = best
        results[:ml_g] = res
    end
    if haskey(param_grids, :ml_m)
        best, res = tune_learner(m.ml_m, X, d, param_grids[:ml_m];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=true)
        m.ml_m = best
        results[:ml_m] = res
    end
    if haskey(param_grids, :ml_pi)
        Xd = hcat(X, d)
        best, res = tune_learner(m.ml_pi, Xd, s, param_grids[:ml_pi];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=true)
        m.ml_pi = best
        results[:ml_pi] = res
    end
    m.fitted = false
    m.boot = nothing
    return results
end

"""
    tune!(m::DoubleMLDID; param_grids, ...)

Tune two-period DID nuisances. Keys: `:ml_g`, `:ml_m`.
"""
function tune!(m::DoubleMLDID;
               param_grids::AbstractDict,
               n_folds_tune::Int=5,
               search_mode::Symbol=:grid,
               n_iter::Int=10,
               rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    results = Dict{Symbol,TuneResult}()
    if haskey(param_grids, :ml_g)
        best, res = tune_learner(m.ml_g, X, y, param_grids[:ml_g];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=false)
        m.ml_g = best
        results[:ml_g] = res
    end
    if haskey(param_grids, :ml_m) && m.ml_m !== nothing
        best, res = tune_learner(m.ml_m, X, d, param_grids[:ml_m];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=true)
        m.ml_m = best
        results[:ml_m] = res
    end
    m.fitted = false
    m.boot = nothing
    return results
end

"""
    tune!(m::DoubleMLPLPR; param_grids, ...)

Tune PLPR nuisances on the transformed panel. Keys: `:ml_l`, `:ml_m`, `:ml_g`.
"""
function tune!(m::DoubleMLPLPR;
               param_grids::AbstractDict,
               n_folds_tune::Int=5,
               search_mode::Symbol=:grid,
               n_iter::Int=10,
               rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    td, _, _ = _transform_plpr(m.data, m.approach)
    X, y, d = td.x, td.y, td.d
    results = Dict{Symbol,TuneResult}()
    if haskey(param_grids, :ml_l)
        best, res = tune_learner(m.ml_l, X, y, param_grids[:ml_l];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=false)
        m.ml_l = best
        results[:ml_l] = res
    end
    if haskey(param_grids, :ml_m)
        best, res = tune_learner(m.ml_m, X, d, param_grids[:ml_m];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng,
                                 classifier=is_classifier(m.ml_m))
        m.ml_m = best
        results[:ml_m] = res
    end
    if haskey(param_grids, :ml_g) && m.ml_g !== nothing
        folds = make_folds(n_obs(td), n_folds_tune; rng=rng)
        ℓ̂ = cross_fit_predict(m.ml_l, X, y, folds)
        m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=is_classifier(m.ml_m))
        v = d .- m̂; u = y .- ℓ̂
        θ0 = sum(v .* u) / max(sum(v .* v), eps())
        best, res = tune_learner(m.ml_g, X, y .- θ0 .* d, param_grids[:ml_g];
                                 n_folds=n_folds_tune, search_mode=search_mode,
                                 n_iter=n_iter, rng=rng, classifier=false)
        m.ml_g = best
        results[:ml_g] = res
    end
    m.fitted = false
    m.boot = nothing
    return results
end
