"""
Abstract base for DoubleML estimators (Python `doubleml.DoubleML` analogue).
"""
abstract type AbstractDoubleML end

# Common result fields expected on concrete types after fit!:
#   coef, se, all_coef, all_se, psi, psi_deriv, boot, fitted, rng, treat_names

# Forward-declared; defined in bootstrap.jl
mutable struct BootstrapResult
    method::String
    n_rep_boot::Int
    boot_t_stat::Array{Float64,3}
end

function t_stat(m::AbstractDoubleML)
    m.fitted || error("Call fit! first")
    return m.coef ./ m.se
end

function pval(m::AbstractDoubleML)
    t = t_stat(m)
    return 2 .* cdf.(Normal(), -abs.(t))
end

"""
    confint(m; level=0.95, joint=false)

Confidence intervals. Pointwise uses normal critical values; `joint=true`
uses the max-|t| multiplier bootstrap (call [`bootstrap!`](@ref) first).
"""
function confint(m::AbstractDoubleML; level::Real=0.95, joint::Bool=false)
    m.fitted || error("Call fit! first")
    (0 < level < 1) || throw(ArgumentError("level must be in (0,1)"))

    n_coef = length(m.coef)
    n_rep = size(m.all_coef, 2)

    if joint
        (m.boot === nothing) && error("Apply bootstrap! before confint(joint=true)")
        # max |t| over coefficients, per bootstrap draw and rep → quantiles per rep
        # boot_t_stat: (n_rep_boot, n_coefs, n_rep)
        max_abs = maximum(abs.(m.boot.boot_t_stat); dims=2)  # (n_rep_boot, 1, n_rep)
        crit_reps = [quantile(vec(max_abs[:, 1, r]), level) for r in 1:n_rep]
        crit = median(crit_reps)
    else
        α = 1 - level
        z = quantile(Normal(), 1 - α / 2)
        crit = z
    end

    # Use the same repeated-split aggregation as coef/se.  In particular,
    # m.se includes between-split variability when n_rep > 1.
    lower = m.coef .- crit .* m.se
    upper = m.coef .+ crit .* m.se

    return DataFrame(
        treatment = m.treat_names,
        lower = lower,
        upper = upper,
        level = fill(Float64(level), n_coef),
        joint = fill(joint, n_coef),
    )
end

"""
    summary_table(m::AbstractDoubleML; level=0.95)

Summary table of coefficient estimates (coef, se, t, p, CI).

Named `summary_table` (not `summary`) to avoid clashing with `Base.summary`
and `StatsBase` when those packages are in scope.
"""
function summary_table(m::AbstractDoubleML; level::Real=0.95)
    m.fitted || error("Call fit! first")
    ci = confint(m; level=level)
    return DataFrame(
        treatment = m.treat_names,
        coef = m.coef,
        std_err = m.se,
        t = t_stat(m),
        pvalue = pval(m),
        ci_lower = ci.lower,
        ci_upper = ci.upper,
    )
end

# Convenience alias used like Python's `.summary` when unambiguous
const dml_summary = summary_table

"""
    set_sample_splitting!(m, smpls)

Set external sample splits (Python `set_sample_splitting`).
`smpls` is a vector of length `n_rep`, each entry a vector of
`(train=..., test=...)` fold named tuples covering `1:n`.
For clustered models, pass the matching `smpls_cluster` keyword returned by
`init_sample_splitting`; ordinary observation folds are not sufficient for
cluster-robust variance estimation.
"""
function _validate_sample_splitting(smpls, n::Int, n_rep::Int)
    length(smpls) == n_rep ||
        throw(ArgumentError("smpls length $(length(smpls)) must equal n_rep=$n_rep"))
    for (r, folds) in enumerate(smpls)
        isempty(folds) && throw(ArgumentError("smpls[$r] must contain at least one fold"))
        seen = zeros(Int, n)
        for (k, fold) in enumerate(folds)
            hasproperty(fold, :train) && hasproperty(fold, :test) ||
                throw(ArgumentError("smpls[$r][$k] must have train and test fields"))
            train = collect(fold.train)
            test = collect(fold.test)
            all(i -> 1 <= i <= n, train) || throw(ArgumentError("train indices out of bounds in smpls[$r][$k]"))
            all(i -> 1 <= i <= n, test) || throw(ArgumentError("test indices out of bounds in smpls[$r][$k]"))
            isempty(intersect(train, test)) ||
                throw(ArgumentError("train and test overlap in smpls[$r][$k]"))
            seen[test] .+= 1
        end
        all(seen .== 1) ||
            throw(ArgumentError("test folds in smpls[$r] must cover each observation exactly once"))
    end
    return nothing
end

function _validate_cluster_sample_splitting(smpls_cluster, smpls, n_rep::Int)
    length(smpls_cluster) == n_rep ||
        throw(ArgumentError("smpls_cluster length must equal n_rep=$n_rep"))
    for r in 1:n_rep
        length(smpls_cluster[r]) == length(smpls[r]) ||
            throw(ArgumentError("smpls_cluster[$r] must match the number of observation folds"))
        for (k, fold) in enumerate(smpls_cluster[r])
            hasproperty(fold, :test_ids) && hasproperty(fold, :train_ids) ||
                throw(ArgumentError("smpls_cluster[$r][$k] must have train_ids and test_ids"))
        end
    end
    return nothing
end

function set_sample_splitting!(m::AbstractDoubleML, smpls; smpls_cluster=nothing)
    hasproperty(m, :smpls) || error("Model does not support sample splitting")
    _validate_sample_splitting(smpls, n_obs(m.data), m.n_rep)
    if hasproperty(m, :smpls_cluster)
        is_cl = hasproperty(m, :is_cluster_data) && m.is_cluster_data === true
        if is_cl
            smpls_cluster === nothing && throw(ArgumentError(
                "clustered models require matching smpls_cluster when setting sample splits"))
            _validate_cluster_sample_splitting(smpls_cluster, smpls, m.n_rep)
        elseif smpls_cluster !== nothing
            throw(ArgumentError("smpls_cluster is only valid for clustered models"))
        end
        m.smpls_cluster = smpls_cluster
    elseif smpls_cluster !== nothing
        throw(ArgumentError("model does not support cluster sample splitting"))
    end
    m.smpls = smpls
    m.fitted = false
    return m
end

"""Use external predictions without evaluating the fallback fit eagerly."""
function _external_or_fit(fallback::Function, external, key::AbstractString, rep::Int, n::Int)
    pred = _apply_external_pred(external, key, rep, n)
    return pred === nothing ? fallback() : pred
end

"""
    set_ml_nuisance_params!(m, learner, treat_var, params)

Set hyperparameters for a nuisance learner (Python `set_ml_nuisance_params`).

# Arguments
- `learner`: e.g. `"ml_l"`, `"ml_m"`, `"ml_g"`, `"ml_g0"`, `"ml_pi"`
- `treat_var`: treatment name (usually `m.data.d_col` or `m.data.d_cols[j]`)
- `params`: `Dict` of keyword params applied via `set_params!` to clones at fit time,
  or nested `Vector` of length `n_rep` of vectors of length `n_folds` for fold-specific params.

Stored on `m.ml_params` when the model has that field.
"""
function set_ml_nuisance_params!(m::AbstractDoubleML, learner::AbstractString,
                                 treat_var::AbstractString, params)
    hasproperty(m, :ml_params) || error("Model does not support ml_params storage")
    key = String(learner)
    tv = String(treat_var)
    if !haskey(m.ml_params, key)
        m.ml_params[key] = Dict{String,Any}()
    end
    m.ml_params[key][tv] = params
    m.fitted = false
    return m
end

"""Resolve params for one fold: returns NamedTuple/kwargs dict or nothing."""
function _params_for_fold(m::AbstractDoubleML, learner::AbstractString, treat_var::AbstractString,
                          rep::Int, fold::Int)
    hasproperty(m, :ml_params) || return nothing
    !haskey(m.ml_params, learner) && return nothing
    d = m.ml_params[learner]
    !haskey(d, treat_var) && return nothing
    p = d[treat_var]
    p === nothing && return nothing
    if p isa AbstractDict
        return p
    elseif p isa AbstractVector
        # n_rep × n_folds nested
        length(p) >= rep || return nothing
        inner = p[rep]
        inner isa AbstractVector || return inner
        length(inner) >= fold || return nothing
        return inner[fold]
    else
        return p
    end
end

function _clone_with_params(learner, params)
    m = clone(learner)
    if params !== nothing && params isa AbstractDict && !isempty(params)
        kwargs = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(params))
        set_params!(m; kwargs...)
    end
    return m
end

"""
    _learner_with_params(m, base, learner_name, treat_var)

Return base learner with global (dict) params applied from `m.ml_params`.
"""
function _learner_with_params(m::AbstractDoubleML, base, learner_name::AbstractString,
                              treat_var::AbstractString)
    hasproperty(m, :ml_params) || return base
    !haskey(m.ml_params, learner_name) && return base
    d = m.ml_params[learner_name]
    !haskey(d, treat_var) && return base
    raw = d[treat_var]
    raw isa AbstractDict || return base
    return _clone_with_params(base, raw)
end

"""
Cross-fit predictions, optionally applying fold-specific params and returning fitted models.

`params_factory(fold_idx) -> Dict or nothing` for fold-specific hyperparameters.
"""
function cross_fit_predict_store(learner, X::AbstractMatrix, y::AbstractVector, folds;
                                 classifier::Bool=false,
                                 params_factory=nothing,
                                 store_models::Bool=false)
    return _cross_fit_one(learner, X, y, folds;
                          classifier=classifier,
                          params_factory=params_factory,
                          store_models=store_models)
end

"""Build fold-params factory from model ml_params for one learner/treat/rep."""
function _fold_params_factory(m::AbstractDoubleML, learner::AbstractString,
                              treat_var::AbstractString, rep::Int)
    hasproperty(m, :ml_params) || return nothing
    !haskey(m.ml_params, learner) && return nothing
    d = m.ml_params[learner]
    !haskey(d, treat_var) && return nothing
    raw = d[treat_var]
    if raw isa AbstractDict
        return _ -> raw
    elseif raw isa AbstractVector && length(raw) >= rep
        inner = raw[rep]
        if inner isa AbstractVector
            return fold -> (length(inner) >= fold ? inner[fold] : nothing)
        end
        return _ -> inner
    end
    return nothing
end

"""
    evaluate_learners(m; metric=rmse) -> Dict

Cross-fitted nuisance prediction quality (Python `evaluate_learners`).
Uses stored `predictions` from `fit!(...; store_predictions=true)`.

# Returns
Dict of learner name → mean metric over available non-NaN entries (averaged across reps).
Default metric is RMSE for continuous targets; for probability predictions still RMSE vs labels when recoverable.
"""
function evaluate_learners(m::AbstractDoubleML; metric::Function=_default_rmse)
    m.fitted || error("Call fit! first")
    hasproperty(m, :predictions) || error("Model has no predictions field")
    isempty(m.predictions) && error("No stored predictions; re-fit with store_predictions=true")
    data = m.data
    out = Dict{String,Float64}()
    for (name, pred) in m.predictions
        pred isa AbstractMatrix || continue
        n, n_rep = size(pred)
        # choose a target vector by learner name convention
        ytrue = _nuisance_target(data, String(name))
        ytrue === nothing && continue
        scores = Float64[]
        for r in 1:n_rep
            p = @view pred[:, r]
            mask = .!isnan.(p) .& .!isnan.(ytrue)
            count(mask) == 0 && continue
            push!(scores, metric(ytrue[mask], p[mask]))
        end
        isempty(scores) || (out[String(name)] = mean(scores))
    end
    return out
end

_default_rmse(y, yhat) = sqrt(mean((y .- yhat) .^ 2))

function _nuisance_target(data::DoubleMLData, name::AbstractString)
    n = lowercase(name)
    if occursin("ml_l", n) || n in ("ml_g", "ml_g0", "ml_g1")
        return data.y
    elseif n == "ml_m" || startswith(n, "ml_m")
        # IIVM/PLIV: m often models Z; PLR/IRM: D
        if n_instr(data) >= 1
            return vec(@view data.z[:, 1])
        end
        return data.d
    elseif n in ("ml_r", "ml_r0", "ml_r1") || startswith(n, "ml_r")
        return data.d
    elseif n == "ml_pi"
        return data.s === nothing ? nothing : data.s
    else
        return nothing
    end
end

"""
    _apply_external_pred(external, key, rep, n) -> Union{Nothing,Vector}

Look up external predictions for one learner and rep.
`external` is `Dict` with keys like `"ml_l"` → `n × n_rep` matrix, or
nested `Dict(treat_name => Dict(learner => matrix))` (first treat used if nested).
"""
function _apply_external_pred(external, key::AbstractString, rep::Int, n::Int)
    external === nothing && return nothing
    mat = nothing
    if haskey(external, key)
        mat = external[key]
    elseif haskey(external, Symbol(key))
        mat = external[Symbol(key)]
    else
        # nested by treatment: use first treatment dict
        for v in values(external)
            if v isa AbstractDict && (haskey(v, key) || haskey(v, Symbol(key)))
                mat = haskey(v, key) ? v[key] : v[Symbol(key)]
                break
            end
        end
    end
    mat === nothing && return nothing
    if mat isa AbstractVector
        length(mat) == n || throw(DimensionMismatch("external $key length"))
        return Float64.(mat)
    elseif mat isa AbstractMatrix
        size(mat, 1) == n || throw(DimensionMismatch("external $key rows"))
        size(mat, 2) >= rep || throw(ArgumentError("external $key missing rep $rep"))
        return Float64.(@view mat[:, rep])
    else
        throw(ArgumentError("external $key must be Vector or Matrix"))
    end
end

"""
    p_adjust(m; method=:holm) -> DataFrame

Multiple-testing adjusted p-values for multi-parameter models.
Methods: `:holm`, `:bonferroni`, `:romano_wolf` (needs `bootstrap!`).
"""
function p_adjust(m::AbstractDoubleML; method::Symbol=:holm)
    m.fitted || error("Call fit! first")
    raw = pval(m)
    n = length(raw)
    if method === :bonferroni
        adj = min.(1.0, raw .* n)
    elseif method === :holm
        ord = sortperm(raw)
        adj = similar(raw)
        for (rank, i) in enumerate(ord)
            adj[i] = min(1.0, raw[i] * (n - rank + 1))
        end
        for k in 2:n
            adj[ord[k]] = max(adj[ord[k]], adj[ord[k - 1]])
        end
    elseif method === :romano_wolf
        hasproperty(m, :boot) || error("bootstrap! required for Romano–Wolf")
        m.boot === nothing && error("Apply bootstrap! before p_adjust(:romano_wolf)")
        t0 = abs.(t_stat(m))
        boot = abs.(m.boot.boot_t_stat)
        boot_med = mapslices(median, boot; dims=3)[:, :, 1]
        ord = sortperm(t0; rev=true)
        adj = ones(n)
        for (step, j) in enumerate(ord)
            remain = ord[step:end]
            max_boot = vec(maximum(boot_med[:, remain]; dims=2))
            adj[j] = mean(max_boot .>= t0[j])
        end
        for step in 2:n
            adj[ord[step]] = max(adj[ord[step]], adj[ord[step - 1]])
        end
    else
        throw(ArgumentError("method must be :holm, :bonferroni, or :romano_wolf"))
    end
    return DataFrame(parameter=m.treat_names, pvalue=raw, pvalue_adjusted=adj,
                     method=fill(String(method), n))
end

function Base.show(io::IO, m::AbstractDoubleML)
    name = string(typeof(m).name.name)
    if !m.fitted
        print(io, "$name (not fitted)")
    else
        print(io, "$name(coef=$(round.(m.coef; digits=4)), se=$(round.(m.se; digits=4)))")
    end
end

# ---- Score validation (Python `_check_score`, allow callables) ----

"""
Validate `score`: either a string in `valid` or a callable (if `allow_callable`).

Callable scores must return `(psi_a, psi_b)` linear score elements with
signature depending on the model (see PLR / IRM docs).
"""
function check_score(score, valid::Tuple{Vararg{AbstractString}}; allow_callable::Bool=true)
    if score isa AbstractString
        s = String(score)
        s in valid || throw(ArgumentError("score must be one of $valid (got \"$s\")"))
        return s
    elseif allow_callable && score isa Function
        return score
    else
        throw(ArgumentError(
            "score must be one of $valid" *
            (allow_callable ? " or a Function returning (psi_a, psi_b)" : "") *
            "; got $(typeof(score))"))
    end
end

is_callable_score(score) = score isa Function

# ---- IPW helpers (shared by IRM / IIVM / PQ / …) ----

_clip_ps(m, ε) = clamp.(m, ε, 1 - ε)

"""
Normalize inverse-probability weights so E[D/m] = E[(1−D)/(1−m)] = 1
(Python `_normalize_ipw`).
"""
function _normalize_ipw(propensity::AbstractVector, treatment::AbstractVector)
    p = Float64.(propensity)
    d = Float64.(treatment)
    p = _clip_ps(p, 1e-8)
    mean_t1 = mean(d ./ p)
    mean_t0 = mean((1 .- d) ./ (1 .- p))
    return d .* (p .* mean_t1) .+ (1 .- d) .* (1 .- (1 .- p) .* mean_t0)
end

# ---- score helpers (linear scores ψ = ψ_a θ + ψ_b) ----

"""
Estimate θ from linear score elements: θ = -mean(ψ_b) / mean(ψ_a).
"""
function est_coef_linear(psi_a::AbstractVector, psi_b::AbstractVector)
    ma = mean(psi_a)
    abs(ma) < 1e-14 && error("Degenerate score: mean(ψ_a) ≈ 0")
    return -mean(psi_b) / ma
end

"""
Standard error for linear DML score (iid case):

    se = sqrt( mean(ψ²) / mean(ψ_a)² / n )

where ψ_i = ψ_a,i * θ + ψ_b,i.
"""
function se_linear(psi_a::AbstractVector, psi_b::AbstractVector, θ::Real)
    n = length(psi_a)
    ψ = psi_a .* θ .+ psi_b
    J = mean(psi_a)           # ∂/∂θ E[ψ] = E[ψ_a]
    σ2 = mean(ψ .^ 2)
    return sqrt(σ2 / (J^2) / n)
end

"""
Aggregate coefficients across repeated sample splits (median of coefs,
mean of squared SEs for variance).
Python DoubleML uses mean of coefs and a specific SE aggregation;
we use the same mean-aggregation as the Python package's default.
"""
function aggregate_reps(all_coef::AbstractMatrix, all_se::AbstractMatrix)
    # all_* : n_coefs × n_rep
    coef = vec(mean(all_coef; dims=2))
    # SE aggregation: sqrt(mean(se²) + sample var of coefs)  — conservative
    # Python uses a refined formula; for n_rep=1 this equals se.
    n_rep = size(all_coef, 2)
    if n_rep == 1
        se = vec(all_se)
    else
        se2 = vec(mean(all_se .^ 2; dims=2))
        # add between-split variability
        if n_rep > 1
            between = vec(var(all_coef; dims=2, corrected=true))
            se = sqrt.(se2 .+ between)
        else
            se = sqrt.(se2)
        end
    end
    return coef, se
end
