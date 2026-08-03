# Repeated cross-section DiD (two periods) — Python DoubleMLDIDCS / Chang (2020)
#
# Data: each row is a unit observed at one time. Need:
#   y, d (group G), t ∈ {0,1} or two distinct times, x
# Target: ATT with observational or experimental score using four regressions
# g_{d,t} = E[Y | D=d, T=t, X] and propensity m = P(D=1|X).

"""
    DoubleMLDIDCS

Repeated cross-section difference-in-differences (two periods).

# Data
[`DoubleMLData`](@ref) with `t` (two periods) and binary group `d`.
Does **not** require `id` (each row is an independent draw).

# Scores
- `"observational"` — conditional parallel trends (needs `ml_m`)
- `"experimental"` — random treatment assignment
"""
mutable struct DoubleMLDIDCS <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    n_folds::Int
    n_rep::Int
    score::String
    in_sample_normalization::Bool
    trimming_threshold::Float64
    t0::Float64
    t1::Float64
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
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLDIDCS(data::DoubleMLData, ml_g, ml_m=nothing;
                       n_folds::Int=5,
                       n_rep::Int=1,
                       score::AbstractString="observational",
                       in_sample_normalization::Bool=true,
                       trimming_threshold::Real=1e-2,
                       draw_sample_splitting::Bool=true,
                       rng::AbstractRNG=Random.default_rng())
    data.t === nothing && throw(ArgumentError("DIDCS requires time variable t in data"))
    sc = String(score)
    sc in ("observational", "experimental") ||
        throw(ArgumentError("score must be observational or experimental"))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) || throw(ArgumentError("binary D required"))
    ts = sort(unique(Float64.(data.t)))
    length(ts) == 2 || throw(ArgumentError("DIDCS requires exactly two time periods"))
    sc == "observational" && ml_m === nothing && throw(ArgumentError("ml_m required"))

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    return DoubleMLDIDCS(
        data, ml_g, ml_m, n_folds, n_rep, sc, in_sample_normalization,
        Float64(trimming_threshold), ts[1], ts[2], smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Any}(), [data.d_col],
        nothing, false, rng,
    )
end

function _cross_fit_g_dt(ml_g, X, y, d, t, dlev, tlev, folds)
    n = size(X, 1)
    g = fill(NaN, n)
    mask = (d .== dlev) .& (t .== tlev)
    for (train, test) in folds
        tr = train[mask[train]]
        isempty(tr) && error("Empty cell D=$dlev,T=$tlev in fold")
        m = clone(ml_g)
        fit!(m, X[tr, :], y[tr])
        g[test] = predict(m, X[test, :])
    end
    return g
end

function _didcs_score(y, d, t, t0, t1, g00, g01, g10, g11, m_hat, score, in_sample)
    n = length(y)
    p = mean(d)
    λ = mean(t .== t1)  # share post
    ind_t1 = t .== t1
    ind_t0 = t .== t0

    if score == "observational"
        if in_sample
            wa = d ./ mean(d)
        else
            wa = d ./ p
        end
        # residualized weights for control outcomes
        if in_sample
            prop_w = (1 .- d) .* (m_hat ./ (1 .- m_hat))
            w_c = prop_w ./ mean(prop_w)
            # Chang/SZ style observational RCS
            psi_b = wa .* (ind_t1 .* (y .- g11) ./ λ .- ind_t0 .* (y .- g10) ./ (1 - λ) .+ (g11 .- g10)) .-
                    w_c .* (ind_t1 .* (y .- g01) ./ λ .- ind_t0 .* (y .- g00) ./ (1 - λ) .+ (g01 .- g00))
            psi_a = -wa
        else
            psi_b = (d ./ p) .* (g11 .- g10) .-
                    ((1 .- d) .* m_hat ./ ((1 .- m_hat) .* p)) .* (g01 .- g00) .+
                    d .* (ind_t1 .* (y .- g11) ./ (p * λ) .- ind_t0 .* (y .- g10) ./ (p * (1 - λ))) .-
                    (1 .- d) .* m_hat ./ (p .* (1 .- m_hat)) .*
                    (ind_t1 .* (y .- g01) ./ λ .- ind_t0 .* (y .- g00) ./ (1 - λ))
            psi_a = fill(-1.0, n) .* (d ./ p)
        end
    else
        # experimental
        wa = ones(n)
        psi_a = -wa
        if in_sample
            wt = d ./ mean(d)
            wc = (1 .- d) ./ mean(1 .- d)
        else
            wt = d ./ p
            wc = (1 .- d) ./ (1 - p)
        end
        psi_b = wt .* (ind_t1 .* (y .- g11) ./ λ .- ind_t0 .* (y .- g10) ./ (1 - λ) .+ (g11 .- g10)) .-
                wc .* (ind_t1 .* (y .- g01) ./ λ .- ind_t0 .* (y .- g00) ./ (1 - λ) .+ (g01 .- g00))
    end
    return psi_a, psi_b
end

function fit!(m::DoubleMLDIDCS; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    t = Float64.(data.t)
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.trimming_threshold
    t0, t1 = m.t0, m.t1

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)

    for r in 1:n_rep
        folds = m.smpls[r]
        g00 = _cross_fit_g_dt(m.ml_g, X, y, d, t, 0.0, t0, folds)
        g01 = _cross_fit_g_dt(m.ml_g, X, y, d, t, 0.0, t1, folds)
        g10 = _cross_fit_g_dt(m.ml_g, X, y, d, t, 1.0, t0, folds)
        g11 = _cross_fit_g_dt(m.ml_g, X, y, d, t, 1.0, t1, folds)
        if m.score == "observational"
            m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=true)
            m̂ = clamp.(m̂, ε, 1 - ε)
        else
            m̂ = fill(mean(d), n)
        end
        psi_a, psi_b = _didcs_score(y, d, t, t0, t1, g00, g01, g10, g11, m̂, m.score, m.in_sample_normalization)
        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)
        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, r, 1] = psi_a
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    m.fitted = true
    return m
end
