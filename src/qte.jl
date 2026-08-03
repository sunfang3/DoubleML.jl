# Quantile / tail treatment effects
# QTE(τ) = θ_τ(1) − θ_τ(0) via paired PQ / LPQ / CVaR models.
# Mirrors Python doubleml.DoubleMLQTE with score ∈ {"PQ","LPQ","CVaR"}.

"""
    DoubleMLQTE

Double machine learning for treatment effects on distributional parameters.

For each quantile `τ` in `quantiles`:

| `score` | Parameter | Building block |
|---------|-----------|----------------|
| `"PQ"` (default) | QTE = θ_τ(1)−θ_τ(0) | [`DoubleMLPQ`](@ref) |
| `"LPQ"` | local QTE (compliers) | [`DoubleMLLPQ`](@ref) (needs Z) |
| `"CVaR"` | CVaR treatment effect | [`DoubleMLCVAR`](@ref) |

# Example
```julia
qte = DoubleMLQTE(data, ml_g, ml_m; quantiles=[0.25, 0.5, 0.75], score="PQ")
fit!(qte)
summary_table(qte)
```
"""
mutable struct DoubleMLQTE <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    quantiles::Vector{Float64}
    score::String
    n_folds::Int
    n_rep::Int
    trimming_threshold::Float64
    ps_processor::PSProcessor
    normalize_ipw::Bool
    smpls::Vector
    coef::Vector{Float64}
    se::Vector{Float64}
    all_coef::Matrix{Float64}
    all_se::Matrix{Float64}
    psi::Array{Float64,3}
    psi_deriv::Array{Float64,3}
    predictions::Dict{String,Any}
    treat_names::Vector{String}
    boot::Union{Nothing,BootstrapResult}
    modellist_0::Vector{Any}
    modellist_1::Vector{Any}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLQTE(data::DoubleMLData, ml_g, ml_m;
                     quantiles=0.5,
                     score::AbstractString="PQ",
                     n_folds::Int=5,
                     n_rep::Int=1,
                     trimming_threshold::Real=1e-2,
                     ps_processor::Union{Nothing,PSProcessor}=nothing,
                     normalize_ipw::Bool=true,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    qs = Float64.(quantiles isa Number ? [quantiles] : collect(quantiles))
    all(0 .< qs .< 1) || throw(ArgumentError("all quantiles must be in (0,1)"))
    sc = String(score)
    sc in ("PQ", "LPQ", "CVaR") ||
        throw(ArgumentError("score must be \"PQ\", \"LPQ\", or \"CVaR\""))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLQTE requires binary treatment in {0,1}"))
    if sc == "LPQ"
        n_instr(data) >= 1 ||
            throw(ArgumentError("score=\"LPQ\" requires instrument Z in DoubleMLData"))
    end

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()

    n = n_obs(data)
    n_q = length(qs)
    prefix = sc == "PQ" ? "QTE" : sc == "LPQ" ? "LQTE" : "CVaR-TE"
    names = ["$prefix(τ=$τ)" for τ in qs]

    psp = resolve_ps_processor(ps_processor, trimming_threshold)
    return DoubleMLQTE(
        data, ml_g, ml_m, qs, sc, n_folds, n_rep,
        Float64(trimming_threshold), psp, normalize_ipw, smpls,
        Float64[], Float64[],
        zeros(n_q, n_rep), zeros(n_q, n_rep),
        fill(NaN, n, n_rep, n_q),
        fill(NaN, n, n_rep, n_q),
        Dict{String,Any}(),
        names,
        nothing,
        Any[], Any[],
        false, rng,
    )
end

function _make_pair_models(m::DoubleMLQTE, τ::Float64)
    kwargs = (
        quantile=τ,
        n_folds=m.n_folds,
        n_rep=m.n_rep,
        trimming_threshold=m.trimming_threshold,
        ps_processor=m.ps_processor,
        normalize_ipw=m.normalize_ipw,
        draw_sample_splitting=false,
        rng=copy(m.rng),
    )
    if m.score == "PQ"
        m0 = DoubleMLPQ(m.data, clone(m.ml_g), clone(m.ml_m); treatment=0, kwargs...)
        m1 = DoubleMLPQ(m.data, clone(m.ml_g), clone(m.ml_m); treatment=1, kwargs...)
    elseif m.score == "LPQ"
        m0 = DoubleMLLPQ(m.data, clone(m.ml_g), clone(m.ml_m); treatment=0, kwargs...)
        m1 = DoubleMLLPQ(m.data, clone(m.ml_g), clone(m.ml_m); treatment=1, kwargs...)
    else  # CVaR
        m0 = DoubleMLCVAR(m.data, clone(m.ml_g), clone(m.ml_m); treatment=0, kwargs...)
        m1 = DoubleMLCVAR(m.data, clone(m.ml_g), clone(m.ml_m); treatment=1, kwargs...)
    end
    m0.smpls = m.smpls
    m1.smpls = m.smpls
    return m0, m1
end

function fit!(m::DoubleMLQTE; store_predictions::Bool=true)
    data = m.data
    n = n_obs(data)
    n_rep = m.n_rep
    n_q = length(m.quantiles)

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    m.modellist_0 = Any[]
    m.modellist_1 = Any[]

    all_coef = zeros(n_q, n_rep)
    all_se = zeros(n_q, n_rep)
    psi_arr = fill(NaN, n, n_rep, n_q)
    psi_d_arr = fill(NaN, n, n_rep, n_q)

    for (iq, τ) in enumerate(m.quantiles)
        m0, m1 = _make_pair_models(m, τ)
        fit!(m0; store_predictions=store_predictions)
        fit!(m1; store_predictions=store_predictions)
        push!(m.modellist_0, m0)
        push!(m.modellist_1, m1)

        for r in 1:n_rep
            θ0 = m0.all_coef[1, r]
            θ1 = m1.all_coef[1, r]
            all_coef[iq, r] = θ1 - θ0
            J0 = mean(@view m0.psi_deriv[:, r, 1])
            J1 = mean(@view m1.psi_deriv[:, r, 1])
            abs(J0) < 1e-14 && error("Degenerate J0 for τ=$τ rep=$r")
            abs(J1) < 1e-14 && error("Degenerate J1 for τ=$τ rep=$r")
            IF = @view(m1.psi[:, r, 1]) ./ J1 .- @view(m0.psi[:, r, 1]) ./ J0
            all_se[iq, r] = sqrt(mean(IF .^ 2) / n)
            psi_arr[:, r, iq] = IF
            psi_d_arr[:, r, iq] .= 1.0
        end
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef
    m.se = se
    m.all_coef = all_coef
    m.all_se = all_se
    m.psi = psi_arr
    m.psi_deriv = psi_d_arr
    m.boot = nothing
    m.fitted = true
    return m
end
