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
        crit = [quantile(vec(max_abs[:, 1, r]), level) for r in 1:n_rep]
    else
        α = 1 - level
        z = quantile(Normal(), 1 - α / 2)
        crit = fill(z, n_rep)
    end

    # CI per rep then median-aggregate (Python DoubleML)
    lower_reps = m.all_coef .- m.all_se .* reshape(crit, 1, n_rep)
    upper_reps = m.all_coef .+ m.all_se .* reshape(crit, 1, n_rep)
    lower = vec(median(lower_reps; dims=2))
    upper = vec(median(upper_reps; dims=2))

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
"""
function set_sample_splitting!(m::AbstractDoubleML, smpls)
    hasproperty(m, :smpls) || error("Model does not support sample splitting")
    length(smpls) == m.n_rep ||
        throw(ArgumentError("smpls length $(length(smpls)) must equal n_rep=$(m.n_rep)"))
    m.smpls = smpls
    m.fitted = false
    return m
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
