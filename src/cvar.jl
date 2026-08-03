# Conditional Value at Risk (CVaR) of potential outcomes — linear DML score
# Python: doubleml.DoubleMLCVAR (Kallus et al.)
#
# With preliminary PQ estimate q̂_τ(d):
#   U = max( q̂, (Y − τ q̂)/(1−τ) )
#   ψ_a = −1,  ψ_b = 1{D=d}(U − ĝ)/m_d + ĝ
#   θ = −mean(ψ_b)/mean(ψ_a)  = CVaR_τ(Y(d))

"""
    DoubleMLCVAR

Double machine learning for the **conditional value at risk** of a potential
outcome `Y(d)` at level `τ` (binary treatment, IRM setting).

`ml_g` must be a **regressor**; `ml_m` a **classifier** (propensity).

# Example
```julia
cvar = DoubleMLCVAR(
    data,
    RidgeLearner(α=0.5),
    LogisticRegressionLearner(α=0.5);
    treatment=1, quantile=0.5, n_folds=5,
)
fit!(cvar)
summary_table(cvar)
```
"""
mutable struct DoubleMLCVAR <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    treatment::Int
    quantile::Float64
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
    predictions::Dict{String,Matrix{Float64}}
    treat_names::Vector{String}
    boot::Union{Nothing,BootstrapResult}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLCVAR(data::DoubleMLData, ml_g, ml_m;
                      treatment::Integer=1,
                      quantile::Real=0.5,
                      n_folds::Int=5,
                      n_rep::Int=1,
                      trimming_threshold::Real=1e-2,
                      normalize_ipw::Bool=true,
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    treatment in (0, 1) || throw(ArgumentError("treatment must be 0 or 1"))
    (0 < quantile < 1) || throw(ArgumentError("quantile must be in (0,1)"))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLCVAR requires binary treatment in {0,1}"))
    is_classifier(ml_m) || @warn "ml_m should be a classifier (propensity)"
    is_classifier(ml_g) && @warn "ml_g should be a regressor for CVaR (got a classifier)"

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    tname = "CVaR(d=$treatment,τ=$(quantile))"
    return DoubleMLCVAR(
        data, ml_g, ml_m, Int(treatment), Float64(quantile),
        n_folds, n_rep, Float64(trimming_threshold), normalize_ipw, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(), [tname],
        nothing, false, rng,
    )
end

"""CVaR transform U = max(q, (Y − τ q)/(1−τ))."""
function _cvar_u(y::AbstractVector, q::Real, τ::Real)
    u1 = fill(Float64(q), length(y))
    u2 = (Float64.(y) .- τ * q) ./ (1 - τ)
    return max.(u1, u2)
end

function _nuisance_est_cvar(ml_g, ml_m, X, y, d, folds, treatment::Int, τ::Float64,
                            trimming::Float64, normalize_ipw::Bool, rng::AbstractRNG)
    n = length(y)
    n_folds = length(folds)
    g_hat = fill(NaN, n)
    m_hat = fill(NaN, n)
    ipw_vec = fill(NaN, n_folds)

    y_treat = y[d .== treatment]
    isempty(y_treat) && error("No observations with D=$treatment")
    # start near empirical CVaR of treated outcomes
    q0 = quantile(y_treat, τ)
    coef_start = mean(y_treat[y_treat .>= q0])
    coef_bounds = (minimum(y), maximum(y))

    for (i_fold, (train, test)) in enumerate(folds)
        tr1, tr2 = _stratified_half(train, d, rng)
        d_tr1 = d[tr1]
        if length(unique(d_tr1)) < 2
            tr1, tr2 = _stratified_half(train, d, rng)
            d_tr1 = d[tr1]
        end
        n_inner = min(n_folds, max(2, length(tr1) ÷ 5))
        m_prelim = _cv_predict_proba_on(ml_m, X, d, tr1, n_inner, rng)
        m_prelim = _clip_ps(m_prelim, trimming)
        if normalize_ipw
            m_prelim = _normalize_ipw(m_prelim, d_tr1)
        end
        m_for_treat = treatment == 1 ? m_prelim : (1 .- m_prelim)
        m_for_treat = _clip_ps(m_for_treat, trimming)

        y_tr1 = y[tr1]
        function ipw_score(θ)
            ind = d_tr1 .== treatment
            return mean(ind .* (y_tr1 .<= θ) ./ m_for_treat .- τ)
        end
        ok, bracket = _get_bracket_guess(ipw_score, coef_start, coef_bounds)
        ipw_est = ok ? _brent_root(ipw_score, bracket[1], bracket[2]) :
                      _minimize_abs(ipw_score, coef_bounds[1], coef_bounds[2])
        ipw_vec[i_fold] = ipw_est

        # g target on tr2 among treated
        d_tr2 = d[tr2]
        treat_mask = d_tr2 .== treatment
        sum(treat_mask) < 2 && error("Too few D=$treatment units in nested train fold for g")
        u_all = _cvar_u(y, ipw_est, τ)
        Xg = X[tr2[treat_mask], :]
        yg = u_all[tr2[treat_mask]]
        g_model = clone(ml_g)
        fit!(g_model, Xg, yg)
        g_hat[test] = predict(g_model, X[test, :])

        m_model = clone(ml_m)
        fit!(m_model, X[train, :], d[train])
        m_hat[test] = predict_proba(m_model, X[test, :])
    end

    m_hat = _clip_ps(m_hat, trimming)
    if normalize_ipw
        m_hat = _normalize_ipw(m_hat, d)
    end
    m_treat = treatment == 1 ? m_hat : (1 .- m_hat)
    m_treat = _clip_ps(m_treat, trimming)
    pq_est = mean(ipw_vec)
    return g_hat, m_treat, pq_est, m_hat
end

function fit!(m::DoubleMLCVAR; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.trimming_threshold
    τ = m.quantile

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    g_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)

    for r in 1:n_rep
        folds = m.smpls[r]
        g_hat, m_treat, pq_est, m_raw = _nuisance_est_cvar(
            m.ml_g, m.ml_m, X, y, d, folds, m.treatment, τ,
            ε, m.normalize_ipw, m.rng,
        )
        u = _cvar_u(y, pq_est, τ)
        ind = d .== m.treatment
        psi_a = fill(-1.0, n)
        psi_b = ind .* (u .- g_hat) ./ m_treat .+ g_hat
        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)
        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, r, 1] = psi_a
        g_preds[:, r] = g_hat
        m_preds[:, r] = m_raw
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_g" => g_preds, "ml_m" => m_preds)
    end
    m.fitted = true
    return m
end
