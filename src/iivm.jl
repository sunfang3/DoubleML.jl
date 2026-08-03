"""
    DoubleMLIIVM

Interactive IV model (binary D, binary Z) targeting LATE.
Mirrors Python `doubleml.DoubleMLIIVM`.

# Subgroups
- `always_takers=true` — estimate `r0 = E[D|X,Z=0]`; if `false`, force `r0=0` (no always-takers)
- `never_takers=true` — estimate `r1 = E[D|X,Z=1]`; if `false`, force `r1=1` (no never-takers)

# Options
- `normalize_ipw` — normalize inverse-probability weights for `m = P(Z=1|X)`
- `subgroups` — optional `Dict(:always_takers=>Bool, :never_takers=>Bool)` (Python API)
"""
mutable struct DoubleMLIIVM <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    ml_r::Any
    n_folds::Int
    n_rep::Int
    score::String
    trimming_threshold::Float64
    always_takers::Bool
    never_takers::Bool
    normalize_ipw::Bool
    smpls::Vector
    smpls_cluster::Union{Nothing,Vector}
    n_folds_per_cluster::Int
    var_scaling::Union{Nothing,Vector{Float64}}
    is_cluster_data::Bool
    cluster_dict::Union{Nothing,NamedTuple}
    coef::Vector{Float64}
    se::Vector{Float64}
    all_coef::Matrix{Float64}
    all_se::Matrix{Float64}
    psi::Array{Float64,3}
    psi_deriv::Array{Float64,3}
    predictions::Dict{String,Matrix{Float64}}
    treat_names::Vector{String}
    boot::Union{Nothing,BootstrapResult}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLIIVM(data::DoubleMLData, ml_g, ml_m, ml_r;
                      n_folds::Int=5,
                      n_rep::Int=1,
                      score::AbstractString="LATE",
                      trimming_threshold::Real=1e-2,
                      always_takers::Bool=true,
                      never_takers::Bool=true,
                      subgroups=nothing,
                      normalize_ipw::Bool=false,
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    score == "LATE" || throw(ArgumentError("score must be \"LATE\""))
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))
    n_treat(data) == 1 || throw(ArgumentError("DoubleMLIIVM supports a single treatment"))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLIIVM requires binary treatment D ∈ {0,1}"))
    data.z === nothing && throw(ArgumentError("DoubleMLIIVM requires a binary instrument"))
    n_instr(data) == 1 || throw(ArgumentError("DoubleMLIIVM requires exactly one binary instrument"))
    z = vec(data.z)
    Set(unique(z)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLIIVM requires binary instrument Z ∈ {0,1}"))

    if subgroups !== nothing
        getsg(k, default) = begin
            if subgroups isa NamedTuple
                haskey(subgroups, k) ? Bool(getfield(subgroups, k)) : default
            elseif subgroups isa AbstractDict
                if haskey(subgroups, k)
                    Bool(subgroups[k])
                elseif haskey(subgroups, String(k))
                    Bool(subgroups[String(k)])
                else
                    default
                end
            else
                default
            end
        end
        always_takers = getsg(:always_takers, always_takers)
        never_takers = getsg(:never_takers, never_takers)
    end

    n = n_obs(data)
    is_cl = is_cluster_data(data)
    smpls, smpls_cluster, n_fpc = if draw_sample_splitting
        init_sample_splitting(data, n_folds, n_rep; rng=rng)
    else
        Vector{Any}(), nothing, n_folds
    end

    return DoubleMLIIVM(
        data, ml_g, ml_m, ml_r, n_folds, n_rep, String(score),
        Float64(trimming_threshold), always_takers, never_takers, normalize_ipw,
        smpls, smpls_cluster, n_fpc, nothing, is_cl, nothing,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [data.d_col], nothing, false, rng,
    )
end

function _cross_fit_conditional(ml, X, y, cond::AbstractVector{Bool}, folds;
                                classifier::Bool=false)
    n = size(X, 1)
    preds = fill(NaN, n)
    for (train, test) in folds
        tr = train[cond[train]]
        isempty(tr) && error("Empty conditional training fold — reduce n_folds or check Z support")
        m = clone(ml)
        fit!(m, X[tr, :], y[tr])
        if classifier || is_classifier(m)
            preds[test] = predict_proba(m, X[test, :])
        else
            preds[test] = predict(m, X[test, :])
        end
    end
    return preds
end

function fit!(m::DoubleMLIIVM; store_predictions::Bool=true,
              external_predictions=nothing)
    data = m.data
    X, y, d = data.x, data.y, data.d
    z = vec(data.z)
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.trimming_threshold
    is_cl = is_cluster_data(data)

    if isempty(m.smpls)
        smpls, smpls_cluster, n_fpc = init_sample_splitting(data, m.n_folds, n_rep; rng=m.rng)
        m.smpls = smpls
        m.smpls_cluster = smpls_cluster
        m.n_folds_per_cluster = n_fpc
        m.is_cluster_data = is_cl
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    var_scaling = fill(Float64(n), 1)
    g0p = fill(NaN, n, n_rep); g1p = fill(NaN, n, n_rep)
    mp = fill(NaN, n, n_rep)
    r0p = fill(NaN, n, n_rep); r1p = fill(NaN, n, n_rep)

    z0 = z .== 0
    z1 = z .== 1

    for rep in 1:n_rep
        folds = m.smpls[rep]
        g0 = something(_apply_external_pred(external_predictions, "ml_g0", rep, n),
                       _cross_fit_conditional(m.ml_g, X, y, z0, folds; classifier=false))
        g1 = something(_apply_external_pred(external_predictions, "ml_g1", rep, n),
                       _cross_fit_conditional(m.ml_g, X, y, z1, folds; classifier=false))
        m̂ = something(_apply_external_pred(external_predictions, "ml_m", rep, n),
                       cross_fit_predict(m.ml_m, X, z, folds; classifier=is_classifier(m.ml_m)))
        m̂ = clamp.(m̂, ε, 1 - ε)
        # normalize IPW using instrument Z (propensity m = P(Z=1|X))
        if m.normalize_ipw
            m̂ = _normalize_ipw(m̂, z)
            m̂ = clamp.(m̂, ε, 1 - ε)
        end

        if m.always_takers
            r0 = something(_apply_external_pred(external_predictions, "ml_r0", rep, n),
                           _cross_fit_conditional(m.ml_r, X, d, z0, folds; classifier=is_classifier(m.ml_r)))
            r0 = clamp.(r0, 0.0, 1.0)
        else
            r0 = zeros(n)
        end
        if m.never_takers
            r1 = something(_apply_external_pred(external_predictions, "ml_r1", rep, n),
                           _cross_fit_conditional(m.ml_r, X, d, z1, folds; classifier=is_classifier(m.ml_r)))
            r1 = clamp.(r1, 0.0, 1.0)
        else
            r1 = ones(n)
        end

        u0 = y .- g0; u1 = y .- g1
        w0 = d .- r0; w1 = d .- r1
        psi_b = (g1 .- g0) .+ z .* u1 ./ m̂ .- (1 .- z) .* u0 ./ (1 .- m̂)
        psi_a = -((r1 .- r0) .+ z .* w1 ./ m̂ .- (1 .- z) .* w0 ./ (1 .- m̂))

        θ = est_coef_linear(psi_a, psi_b)
        se, vsf = se_from_score(psi_a, psi_b, θ;
                                smpls=folds,
                                cluster=is_cl ? data.cluster : nothing,
                                smpls_cluster=is_cl ? m.smpls_cluster[rep] : nothing,
                                n_folds_per_cluster=m.n_folds_per_cluster)
        all_coef[1, rep] = θ
        all_se[1, rep] = se
        psi_arr[:, rep, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, rep, 1] = psi_a
        var_scaling[1] = vsf
        g0p[:, rep] = g0; g1p[:, rep] = g1
        mp[:, rep] = m̂
        r0p[:, rep] = r0; r1p[:, rep] = r1
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.var_scaling = var_scaling
    m.boot = nothing
    if is_cl
        m.cluster_dict = (
            smpls=m.smpls, smpls_cluster=m.smpls_cluster,
            cluster_vars=data.cluster, n_folds_per_cluster=m.n_folds_per_cluster,
        )
    end
    if store_predictions
        m.predictions = Dict(
            "ml_g0" => g0p, "ml_g1" => g1p, "ml_m" => mp,
            "ml_r0" => r0p, "ml_r1" => r1p,
        )
    end
    m.fitted = true
    return m
end
