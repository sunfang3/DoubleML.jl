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
    if search_mode === :optuna
        space = Dict{Symbol,Any}(k => v for (k, v) in param_grid)
        best, ores = tune_learner_optuna(learner, X, y, space;
                                         n_trials=n_iter, n_folds=n_folds, rng=rng,
                                         classifier=classifier)
        return best, TuneResult(ores.best_params, ores.best_score,
                                ores.history_params, ores.history_scores)
    end

    combos = expand_param_grid(param_grid)
    if search_mode === :random
        n_iter = min(n_iter, length(combos))
        combos = combos[randperm(rng, length(combos))[1:n_iter]]
    elseif search_mode !== :grid
        throw(ArgumentError("search_mode must be :grid, :random, or :optuna"))
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

# ---- Optuna-style adaptive search (no Optuna dependency) --------------------

"""
    DMLOptunaResult

Lightweight stand-in for Python `DMLOptunaResult` / Optuna study output.
"""
struct DMLOptunaResult
    best_params::Dict{Symbol,Any}
    best_score::Float64
    n_trials::Int
    history_params::Vector{Dict{Symbol,Any}}
    history_scores::Vector{Float64}
end

function Base.show(io::IO, r::DMLOptunaResult)
    print(io, "DMLOptunaResult(best_score=$(round(r.best_score; digits=5)), ",
          "n_trials=$(r.n_trials), best=$(r.best_params))")
end

"""
Sample one parameter set from a space.

Space entry formats:
- `Vector` — discrete choices (uniform)
- `Tuple{Real,Real}` — continuous uniform on `[lo, hi]`
- `Tuple{Real,Real,Symbol}` — with `:log` or `:linear` scale
"""
function _sample_space(space::AbstractDict{Symbol}, rng::AbstractRNG)
    params = Dict{Symbol,Any}()
    for (k, spec) in space
        if spec isa AbstractVector
            params[k] = spec[rand(rng, 1:length(spec))]
        elseif spec isa Tuple && length(spec) >= 2
            lo, hi = Float64(spec[1]), Float64(spec[2])
            scale = length(spec) >= 3 ? spec[3] : :linear
            if scale === :log
                lo > 0 && hi > 0 || throw(ArgumentError("log scale requires positive bounds for $k"))
                u = rand(rng)
                params[k] = exp(log(lo) + u * (log(hi) - log(lo)))
            else
                params[k] = lo + rand(rng) * (hi - lo)
            end
        else
            throw(ArgumentError("Unsupported space entry for $k: $spec"))
        end
    end
    return params
end

"""
TPE-inspired local refinement: sample around best continuous params, keep discrete.
"""
function _sample_near(best::Dict{Symbol,Any}, space::AbstractDict{Symbol}, rng::AbstractRNG;
                      width::Float64=0.3)
    params = Dict{Symbol,Any}()
    for (k, spec) in space
        if spec isa AbstractVector
            # 70% keep best, 30% redraw
            params[k] = rand(rng) < 0.7 && haskey(best, k) ? best[k] : spec[rand(rng, 1:length(spec))]
        elseif spec isa Tuple
            lo, hi = Float64(spec[1]), Float64(spec[2])
            scale = length(spec) >= 3 ? spec[3] : :linear
            b = Float64(get(best, k, (lo + hi) / 2))
            if scale === :log
                lb, ub = log(lo), log(hi)
                σ = width * (ub - lb)
                cand = exp(clamp(log(max(b, lo)) + σ * randn(rng), lb, ub))
                params[k] = cand
            else
                σ = width * (hi - lo)
                params[k] = clamp(b + σ * randn(rng), lo, hi)
            end
        end
    end
    return params
end

"""
    tune_learner_optuna(learner, X, y, param_space; n_trials=30, n_folds=5, ...)

Optuna-style hyperparameter search **without** the Optuna package:
1. `n_startup` random trials from `param_space`
2. Remaining trials sample near the current best (TPE-lite / local exploit)

# Returns
`(best_learner, DMLOptunaResult)`
"""
function tune_learner_optuna(learner, X::AbstractMatrix, y::AbstractVector,
                             param_space::AbstractDict{Symbol};
                             n_trials::Int=30,
                             n_startup::Int=10,
                             n_folds::Int=5,
                             rng::AbstractRNG=Random.default_rng(),
                             classifier::Bool=is_classifier(learner))
    n_trials >= 1 || throw(ArgumentError("n_trials ≥ 1"))
    n_startup = clamp(n_startup, 1, n_trials)
    folds = make_folds(size(X, 1), n_folds; rng=rng)
    hist_p = Dict{Symbol,Any}[]
    hist_s = Float64[]
    best_score = Inf
    best_params = Dict{Symbol,Any}()
    best_learner = clone(learner)

    for t in 1:n_trials
        params = if t <= n_startup || best_score == Inf
            _sample_space(param_space, rng)
        else
            _sample_near(best_params, param_space, rng)
        end
        cand = clone(learner)
        # cast integer-looking floats for discrete-friendly params like max_depth if needed
        kwargs = Dict{Symbol,Any}(k => (v isa AbstractFloat && abs(v - round(v)) < 1e-10 ? Int(round(v)) : v)
                                  for (k, v) in params)
        sc = Inf
        try
            set_params!(cand; kwargs...)
            sc = cv_score(cand, X, y, folds; classifier=classifier)
        catch
            sc = Inf
        end
        push!(hist_p, params)
        push!(hist_s, sc)
        if sc < best_score
            best_score = sc
            best_params = params
            best_learner = cand
        end
    end
    res = DMLOptunaResult(best_params, best_score, n_trials, hist_p, hist_s)
    return best_learner, res
end

"""
    tune_optuna!(m::DoubleMLPLR; param_spaces, n_trials=30, ...)

Optuna-style nuisance tuning for PLR (Python `tune` + Optuna analogue).

`param_spaces` keys `:ml_l`, `:ml_m`, optionally `:ml_g`.
Each value is a space dict as in [`tune_learner_optuna`](@ref).
"""
function tune_optuna!(m::DoubleMLPLR;
                      param_spaces::AbstractDict,
                      n_trials::Int=30,
                      n_startup::Int=10,
                      n_folds_tune::Int=5,
                      rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y = data.x, data.y
    d = data.d
    results = Dict{Symbol,DMLOptunaResult}()
    if haskey(param_spaces, :ml_l)
        best, res = tune_learner_optuna(m.ml_l, X, y, param_spaces[:ml_l];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=false)
        m.ml_l = best; results[:ml_l] = res
    end
    if haskey(param_spaces, :ml_m)
        best, res = tune_learner_optuna(m.ml_m, X, d, param_spaces[:ml_m];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng,
                                        classifier=is_classifier(m.ml_m))
        m.ml_m = best; results[:ml_m] = res
    end
    if haskey(param_spaces, :ml_g) && m.ml_g !== nothing
        best, res = tune_learner_optuna(m.ml_g, X, y, param_spaces[:ml_g];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=false)
        m.ml_g = best; results[:ml_g] = res
    end
    m.fitted = false
    m.boot = nothing
    return results
end

function tune_optuna!(m::DoubleMLIRM;
                      param_spaces::AbstractDict,
                      n_trials::Int=30,
                      n_startup::Int=10,
                      n_folds_tune::Int=5,
                      rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    results = Dict{Symbol,DMLOptunaResult}()
    if haskey(param_spaces, :ml_g)
        best, res = tune_learner_optuna(m.ml_g, X, y, param_spaces[:ml_g];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=false)
        m.ml_g = best; results[:ml_g] = res
    end
    if haskey(param_spaces, :ml_m)
        best, res = tune_learner_optuna(m.ml_m, X, d, param_spaces[:ml_m];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=true)
        m.ml_m = best; results[:ml_m] = res
    end
    m.fitted = false
    m.boot = nothing
    return results
end


function tune_optuna!(m::DoubleMLPLIV;
                      param_spaces::AbstractDict,
                      n_trials::Int=30,
                      n_startup::Int=10,
                      n_folds_tune::Int=5,
                      rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    Z = data.z
    results = Dict{Symbol,DMLOptunaResult}()
    if m.partial_mode == :partialZ
        XZ = hcat(X, Z)
        if haskey(param_spaces, :ml_r) && m.ml_r !== nothing
            best, res = tune_learner_optuna(m.ml_r, XZ, d, param_spaces[:ml_r];
                                            n_trials=n_trials, n_startup=n_startup,
                                            n_folds=n_folds_tune, rng=rng, classifier=false)
            m.ml_r = best; results[:ml_r] = res
        end
    else
        if haskey(param_spaces, :ml_l) && m.ml_l !== nothing
            best, res = tune_learner_optuna(m.ml_l, X, y, param_spaces[:ml_l];
                                            n_trials=n_trials, n_startup=n_startup,
                                            n_folds=n_folds_tune, rng=rng, classifier=false)
            m.ml_l = best; results[:ml_l] = res
        end
        if m.partial_mode == :partialX
            if haskey(param_spaces, :ml_r) && m.ml_r !== nothing
                best, res = tune_learner_optuna(m.ml_r, X, d, param_spaces[:ml_r];
                                                n_trials=n_trials, n_startup=n_startup,
                                                n_folds=n_folds_tune, rng=rng, classifier=false)
                m.ml_r = best; results[:ml_r] = res
            end
            if haskey(param_spaces, :ml_m) && m.ml_m !== nothing
                z = n_instr(data) == 1 ? vec(Z) : Z[:, 1]
                best, res = tune_learner_optuna(m.ml_m, X, z, param_spaces[:ml_m];
                                                n_trials=n_trials, n_startup=n_startup,
                                                n_folds=n_folds_tune, rng=rng, classifier=false)
                m.ml_m = best; results[:ml_m] = res
            end
            if haskey(param_spaces, :ml_g) && m.ml_g !== nothing
                best, res = tune_learner_optuna(m.ml_g, X, y, param_spaces[:ml_g];
                                                n_trials=n_trials, n_startup=n_startup,
                                                n_folds=n_folds_tune, rng=rng, classifier=false)
                m.ml_g = best; results[:ml_g] = res
            end
        elseif m.partial_mode == :partialXZ
            XZ = hcat(X, Z)
            if haskey(param_spaces, :ml_m) && m.ml_m !== nothing
                best, res = tune_learner_optuna(m.ml_m, XZ, d, param_spaces[:ml_m];
                                                n_trials=n_trials, n_startup=n_startup,
                                                n_folds=n_folds_tune, rng=rng, classifier=false)
                m.ml_m = best; results[:ml_m] = res
            end
            if haskey(param_spaces, :ml_r) && m.ml_r !== nothing
                folds = make_folds(n_obs(data), n_folds_tune; rng=rng)
                mhat = cross_fit_predict(m.ml_m, XZ, d, folds)
                best, res = tune_learner_optuna(m.ml_r, X, mhat, param_spaces[:ml_r];
                                                n_trials=n_trials, n_startup=n_startup,
                                                n_folds=n_folds_tune, rng=rng, classifier=false)
                m.ml_r = best; results[:ml_r] = res
            end
        end
    end
    m.fitted = false
    m.boot = nothing
    return results
end

function tune_optuna!(m::DoubleMLIIVM;
                      param_spaces::AbstractDict,
                      n_trials::Int=30,
                      n_startup::Int=10,
                      n_folds_tune::Int=5,
                      rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    z = vec(data.z)
    results = Dict{Symbol,DMLOptunaResult}()
    if haskey(param_spaces, :ml_g)
        best, res = tune_learner_optuna(m.ml_g, X, y, param_spaces[:ml_g];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=false)
        m.ml_g = best; results[:ml_g] = res
    end
    if haskey(param_spaces, :ml_m)
        best, res = tune_learner_optuna(m.ml_m, X, z, param_spaces[:ml_m];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=true)
        m.ml_m = best; results[:ml_m] = res
    end
    if haskey(param_spaces, :ml_r)
        best, res = tune_learner_optuna(m.ml_r, X, d, param_spaces[:ml_r];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=true)
        m.ml_r = best; results[:ml_r] = res
    end
    m.fitted = false
    m.boot = nothing
    return results
end

function tune_optuna!(m::DoubleMLSSM;
                      param_spaces::AbstractDict,
                      n_trials::Int=30,
                      n_startup::Int=10,
                      n_folds_tune::Int=5,
                      rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d, s = data.x, data.y, data.d, data.s
    results = Dict{Symbol,DMLOptunaResult}()
    if haskey(param_spaces, :ml_g)
        # outcome among selected
        sel = s .== 1
        best, res = tune_learner_optuna(m.ml_g, X[sel, :], y[sel], param_spaces[:ml_g];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=false)
        m.ml_g = best; results[:ml_g] = res
    end
    if haskey(param_spaces, :ml_m)
        best, res = tune_learner_optuna(m.ml_m, X, d, param_spaces[:ml_m];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=true)
        m.ml_m = best; results[:ml_m] = res
    end
    if haskey(param_spaces, :ml_pi)
        Xd = hcat(X, d)
        best, res = tune_learner_optuna(m.ml_pi, Xd, s, param_spaces[:ml_pi];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=true)
        m.ml_pi = best; results[:ml_pi] = res
    end
    m.fitted = false
    m.boot = nothing
    return results
end

function tune_optuna!(m::DoubleMLDID;
                      param_spaces::AbstractDict,
                      n_trials::Int=30,
                      n_startup::Int=10,
                      n_folds_tune::Int=5,
                      rng::Union{Nothing,AbstractRNG}=nothing)
    rng = rng === nothing ? m.rng : rng
    data = m.data
    X, y, d = data.x, data.y, data.d
    results = Dict{Symbol,DMLOptunaResult}()
    if haskey(param_spaces, :ml_g)
        best, res = tune_learner_optuna(m.ml_g, X, y, param_spaces[:ml_g];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=false)
        m.ml_g = best; results[:ml_g] = res
    end
    if haskey(param_spaces, :ml_m) && m.ml_m !== nothing
        best, res = tune_learner_optuna(m.ml_m, X, d, param_spaces[:ml_m];
                                        n_trials=n_trials, n_startup=n_startup,
                                        n_folds=n_folds_tune, rng=rng, classifier=true)
        m.ml_m = best; results[:ml_m] = res
    end
    m.fitted = false
    m.boot = nothing
    return results
end
