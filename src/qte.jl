# Quantile Treatment Effects (QTE)
# QTE(τ) = θ_τ(1) − θ_τ(0) via paired DoubleMLPQ models.
# Mirrors Python doubleml.DoubleMLQTE (score="PQ").

"""
    DoubleMLQTE

Double machine learning for **quantile treatment effects**.

For each quantile `τ` in `quantiles`:

```
QTE(τ) = θ_τ(1) − θ_τ(0)
```

where `θ_τ(d)` is the potential quantile from [`DoubleMLPQ`](@ref).

# Example
```julia
data = make_irm_data(n_obs=1500, dim_x=5, theta=0.5; seed=1)
qte = DoubleMLQTE(
    data,
    LogisticRegressionLearner(α=0.5),
    LogisticRegressionLearner(α=0.5);
    quantiles=[0.25, 0.5, 0.75], n_folds=5,
)
fit!(qte)
summary_table(qte)
```
"""
mutable struct DoubleMLQTE <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    quantiles::Vector{Float64}
    n_folds::Int
    n_rep::Int
    trimming_threshold::Float64
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
    modellist_0::Vector{DoubleMLPQ}
    modellist_1::Vector{DoubleMLPQ}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLQTE(data::DoubleMLData, ml_g, ml_m;
                     quantiles=0.5,
                     n_folds::Int=5,
                     n_rep::Int=1,
                     trimming_threshold::Real=1e-2,
                     normalize_ipw::Bool=true,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    qs = Float64.(quantiles isa Number ? [quantiles] : collect(quantiles))
    all(0 .< qs .< 1) || throw(ArgumentError("all quantiles must be in (0,1)"))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLQTE requires binary treatment in {0,1}"))

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()

    n = n_obs(data)
    n_q = length(qs)
    names = ["QTE(τ=$τ)" for τ in qs]

    # placeholder PQ lists (rebuilt in fit!)
    return DoubleMLQTE(
        data, ml_g, ml_m, qs, n_folds, n_rep,
        Float64(trimming_threshold), normalize_ipw, smpls,
        Float64[], Float64[],
        zeros(n_q, n_rep), zeros(n_q, n_rep),
        fill(NaN, n, n_rep, n_q),
        fill(NaN, n, n_rep, n_q),
        Dict{String,Any}(),
        names,
        nothing,
        DoubleMLPQ[], DoubleMLPQ[],
        false, rng,
    )
end

function fit!(m::DoubleMLQTE; store_predictions::Bool=true)
    data = m.data
    n = n_obs(data)
    n_rep = m.n_rep
    n_q = length(m.quantiles)

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    # allocate PQ models sharing sample splits
    m.modellist_0 = DoubleMLPQ[]
    m.modellist_1 = DoubleMLPQ[]

    all_coef = zeros(n_q, n_rep)
    all_se = zeros(n_q, n_rep)
    psi_arr = fill(NaN, n, n_rep, n_q)
    psi_d_arr = fill(NaN, n, n_rep, n_q)

    for (iq, τ) in enumerate(m.quantiles)
        # fresh clones per quantile so nuisance learners are independent
        pq0 = DoubleMLPQ(
            data, clone(m.ml_g), clone(m.ml_m);
            treatment=0, quantile=τ,
            n_folds=m.n_folds, n_rep=n_rep,
            trimming_threshold=m.trimming_threshold,
            normalize_ipw=m.normalize_ipw,
            draw_sample_splitting=false,
            rng=copy(m.rng),
        )
        pq1 = DoubleMLPQ(
            data, clone(m.ml_g), clone(m.ml_m);
            treatment=1, quantile=τ,
            n_folds=m.n_folds, n_rep=n_rep,
            trimming_threshold=m.trimming_threshold,
            normalize_ipw=m.normalize_ipw,
            draw_sample_splitting=false,
            rng=copy(m.rng),
        )
        # share folds with parent for comparability
        pq0.smpls = m.smpls
        pq1.smpls = m.smpls
        fit!(pq0; store_predictions=store_predictions)
        fit!(pq1; store_predictions=store_predictions)
        push!(m.modellist_0, pq0)
        push!(m.modellist_1, pq1)

        for r in 1:n_rep
            θ0 = pq0.all_coef[1, r]
            θ1 = pq1.all_coef[1, r]
            all_coef[iq, r] = θ1 - θ0
            # influence-function SE: IF = ψ1/J1 − ψ0/J0
            J0 = mean(@view pq0.psi_deriv[:, r, 1])
            J1 = mean(@view pq1.psi_deriv[:, r, 1])
            abs(J0) < 1e-14 && error("Degenerate J0 for τ=$τ rep=$r")
            abs(J1) < 1e-14 && error("Degenerate J1 for τ=$τ rep=$r")
            IF = @view(pq1.psi[:, r, 1]) ./ J1 .- @view(pq0.psi[:, r, 1]) ./ J0
            all_se[iq, r] = sqrt(mean(IF .^ 2) / n)
            # store IF * J_dummy for bootstrap (use J=1 so scaled = IF)
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
