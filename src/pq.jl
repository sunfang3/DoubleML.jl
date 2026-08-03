# Potential Quantiles (PQ) — nonlinear DML
# Chernozhukov et al. / Python doubleml.DoubleMLPQ
#
# Target: θ_τ(d) s.t. P(Y(d) ≤ θ_τ(d)) = τ
# Score: ψ(θ) = 1{D=d} ((1{Y≤θ} − g(X)) / m_d(X)) + g(X) − τ
# with g(X) = P(Y≤θ | X, D=d), m_d(X) = P(D=d | X).

# ---- small numeric utilities ------------------------------------------------
# _clip_ps / _normalize_ipw defined in base.jl

"""Stratified 50/50 split of index vector by binary labels `d`."""
function _stratified_half(idx::AbstractVector{Int}, d::AbstractVector, rng::AbstractRNG)
    i0 = idx[d[idx] .== 0]
    i1 = idx[d[idx] .== 1]
    i0 = Random.shuffle(rng, collect(i0))
    i1 = Random.shuffle(rng, collect(i1))
    n0, n1 = length(i0), length(i1)
    h0, h1 = max(n0 ÷ 2, 1), max(n1 ÷ 2, 0)
    # ensure both halves non-empty when possible
    if n1 >= 2
        h1 = n1 ÷ 2
    elseif n1 == 1
        h1 = 0  # put the single treated in the larger half randomly
    end
    a = sort(vcat(i0[1:h0], i1[1:h1]))
    b = sort(vcat(i0[h0+1:end], i1[h1+1:end]))
    if isempty(a) || isempty(b)
        # fallback: plain half split
        idx_s = Random.shuffle(rng, collect(idx))
        mid = max(length(idx_s) ÷ 2, 1)
        a = sort(idx_s[1:mid])
        b = sort(idx_s[mid+1:end])
    end
    return a, b
end

"""
Find a bracket [a,b] ⊆ bounds around `start` where `f` changes sign.
Returns `(success, (a,b))`.
"""
function _get_bracket_guess(f::Function, start::Real, bounds::Tuple{<:Real,<:Real})
    lo, hi = Float64(bounds[1]), Float64(bounds[2])
    (hi > lo) || return false, (lo, hi)
    max_len = hi - lo
    s0 = Float64(start)
    delta = 0.1
    a, b = lo, hi
    while delta <= 1.0 + 1e-12
        a = max(s0 - delta * max_len / 2, lo)
        b = min(s0 + delta * max_len / 2, hi)
        fa, fb = f(a), f(b)
        if isfinite(fa) && isfinite(fb) && sign(fa) != sign(fb)
            return true, (a, b)
        end
        delta += 0.1
    end
    # last resort: full bounds
    fa, fb = f(lo), f(hi)
    return (isfinite(fa) && isfinite(fb) && sign(fa) != sign(fb)), (lo, hi)
end

"""Brent's method for a root of `f` on bracket [a,b] (sign change required)."""
function _brent_root(f::Function, a::Real, b::Real; tol::Float64=1e-10, maxiter::Int=200)
    a, b = Float64(a), Float64(b)
    fa, fb = f(a), f(b)
    if !(isfinite(fa) && isfinite(fb))
        error("Non-finite score at bracket endpoints")
    end
    if sign(fa) == sign(fb)
        # minimize |f| on [a,b] as fallback (golden section)
        return _minimize_abs(f, a, b)
    end
    if abs(fa) < abs(fb)
        a, b, fa, fb = b, a, fb, fa
    end
    c, fc = a, fa
    d = b - a
    e = d
    for _ in 1:maxiter
        if sign(fb) == sign(fc)
            c, fc = a, fa
            d = b - a
            e = d
        end
        if abs(fc) < abs(fb)
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        end
        tol1 = 2 * eps(Float64) * abs(b) + tol / 2
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 || fb == 0
            return b
        end
        if abs(e) >= tol1 && abs(fa) > abs(fb)
            s = fb / fa
            if a == c
                # secant
                p = 2 * xm * s
                q = 1 - s
            else
                # inverse quadratic
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            end
            if p > 0
                q = -q
            else
                p = -p
            end
            s = e
            e = d
            if 2p < 3 * xm * q - abs(tol1 * q) && p < abs(0.5 * s * q)
                d = p / q
            else
                d = xm
                e = d
            end
        else
            d = xm
            e = d
        end
        a, fa = b, fb
        if abs(d) > tol1
            b += d
        else
            b += copysign(tol1, xm)
        end
        fb = f(b)
    end
    return b
end

function _minimize_abs(f::Function, a::Real, b::Real; n::Int=80)
    # golden-section on |f|
    φ = (sqrt(5) - 1) / 2
    lo, hi = Float64(a), Float64(b)
    c = hi - φ * (hi - lo)
    d = lo + φ * (hi - lo)
    fc, fd = abs(f(c)), abs(f(d))
    for _ in 1:n
        if fc < fd
            hi, d, fd = d, c, fc
            c = hi - φ * (hi - lo)
            fc = abs(f(c))
        else
            lo, c, fc = c, d, fd
            d = lo + φ * (hi - lo)
            fd = abs(f(d))
        end
    end
    return (lo + hi) / 2
end

"""
Weighted Gaussian KDE of residuals `u = y − θ` evaluated as mean(w · K_h(u)),
estimating E[w δ(Y−θ)] (score derivative).
"""
function _weighted_kde_deriv(u::AbstractVector, w::AbstractVector)
    n = length(u)
    n >= 2 || return 0.0
    sw = sum(w)
    sw <= 0 && return 0.0
    # weighted mean / var for Silverman bandwidth
    μ = sum(w .* u) / sw
    v = sum(w .* (u .- μ) .^ 2) / sw
    σ = sqrt(max(v, eps()))
    # effective sample size
    n_eff = sw^2 / max(sum(w .^ 2), eps())
    h = 1.06 * σ * n_eff^(-0.2)
    h = max(h, 1e-6 * (1 + abs(μ)))
    # mean_i [ w_i φ_h(u_i) ]
    invh = 1 / h
    c = invh / sqrt(2π)
    s = 0.0
    @inbounds for i in 1:n
        z = u[i] * invh
        s += w[i] * c * exp(-0.5 * z * z)
    end
    return s / n
end

# ---- PQ score ---------------------------------------------------------------

"""PQ moment at θ given nuisance predictions (m already treatment-specific)."""
function _pq_score_mean(θ, y, d, g, m, treatment::Int, τ::Float64)
    ind = d .== treatment
    return mean(ind .* ((y .<= θ) .- g) ./ m .+ g .- τ)
end

function _pq_score_vec(θ, y, d, g, m, treatment::Int, τ::Float64)
    ind = d .== treatment
    return ind .* ((y .<= θ) .- g) ./ m .+ g .- τ
end

# ---- nuisance estimation with nested CF ------------------------------------

"""
Cross-fit propensity on a subset of indices with given fold partition
(relative indices into `idx`).
"""
function _cv_predict_proba_on(ml, X, y_bin, idx::Vector{Int}, n_folds::Int, rng)
    n_sub = length(idx)
    preds = fill(NaN, n_sub)
    # map local folds
    folds = make_folds(n_sub, min(n_folds, n_sub); rng=rng)
    for (train_loc, test_loc) in folds
        tr = idx[train_loc]
        te_loc = test_loc
        te = idx[te_loc]
        m = clone(ml)
        fit!(m, X[tr, :], y_bin[tr])
        preds[te_loc] = predict_proba(m, X[te, :])
    end
    return preds
end

"""
Nested cross-fitting for PQ nuisances (g, m) for one sample-split partition.
Returns `g_hat`, `m_hat_treat` (already = P(D=treatment|X) adjusted), and mean IPW start.
"""
function _nuisance_est_pq(ml_g, ml_m, X, y, d, folds, treatment::Int, τ::Float64,
                          trimming::Float64, normalize_ipw::Bool, rng::AbstractRNG)
    n = length(y)
    n_folds = length(folds)
    g_hat = fill(NaN, n)
    m_hat = fill(NaN, n)
    ipw_vec = fill(NaN, n_folds)

    y_treat = y[d .== treatment]
    isempty(y_treat) && error("No observations with D=$treatment")
    coef_start = quantile(y_treat, τ)
    coef_bounds = (minimum(y), maximum(y))

    for (i_fold, (train, test)) in enumerate(folds)
        # nested split of train
        tr1, tr2 = _stratified_half(train, d, rng)

        # --- preliminary propensity on tr1 via inner CV ---
        d_tr1 = d[tr1]
        # ensure both classes present
        if length(unique(d_tr1)) < 2
            tr1, tr2 = _stratified_half(train, d, rng)  # retry with new shuffle
            d_tr1 = d[tr1]
        end
        n_inner = min(n_folds, max(2, length(tr1) ÷ 5))
        m_prelim_local = _cv_predict_proba_on(ml_m, X, d, tr1, n_inner, rng)
        m_prelim = _clip_ps(m_prelim_local, trimming)
        if normalize_ipw
            m_prelim = _normalize_ipw(m_prelim, d_tr1)
        end
        # P(D = treatment | X)
        m_for_treat = treatment == 1 ? m_prelim : (1 .- m_prelim)
        m_for_treat = _clip_ps(m_for_treat, trimming)

        y_tr1 = y[tr1]
        d_tr1f = d_tr1
        function ipw_score(θ)
            ind = d_tr1f .== treatment
            return mean(ind .* (y_tr1 .<= θ) ./ m_for_treat .- τ)
        end
        ok, bracket = _get_bracket_guess(ipw_score, coef_start, coef_bounds)
        ipw_est = ok ? _brent_root(ipw_score, bracket[1], bracket[2]) :
                      _minimize_abs(ipw_score, coef_bounds[1], coef_bounds[2])
        ipw_vec[i_fold] = ipw_est

        # --- fit g on tr2 among treated units with label 1{Y ≤ ipw_est} ---
        d_tr2 = d[tr2]
        y_tr2 = y[tr2]
        treat_mask = d_tr2 .== treatment
        sum(treat_mask) < 2 && error("Too few D=$treatment units in nested train fold for g")
        Xg = X[tr2[treat_mask], :]
        yg = Float64.(y_tr2[treat_mask] .<= ipw_est)
        # if constant labels, g is constant
        g_model = clone(ml_g)
        if length(unique(yg)) < 2
            gconst = mean(yg)
            g_hat[test] .= gconst
        else
            fit!(g_model, Xg, yg)
            g_hat[test] = predict_proba(g_model, X[test, :])
        end

        # --- refit propensity on full train ---
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

    g_hat = clamp.(g_hat, 0.0, 1.0)
    ipw_start = mean(ipw_vec)
    return g_hat, m_treat, ipw_start, m_hat
end

function _pq_estimate(y, d, g, m_treat, treatment::Int, τ::Float64, ipw_start::Float64)
    n = length(y)
    bounds = (minimum(y), maximum(y))
    f = θ -> _pq_score_mean(θ, y, d, g, m_treat, treatment, τ)
    ok, bracket = _get_bracket_guess(f, ipw_start, bounds)
    θ = ok ? _brent_root(f, bracket[1], bracket[2]) :
             _minimize_abs(f, bounds[1], bounds[2])

    ψ = _pq_score_vec(θ, y, d, g, m_treat, treatment, τ)
    # score derivative via weighted KDE
    ind = d .== treatment
    w = ind ./ m_treat
    J = _weighted_kde_deriv(y .- θ, w)
    if abs(J) < 1e-10
        # finite-difference fallback
        h = max(1e-4 * (bounds[2] - bounds[1]), 1e-6)
        J = (f(θ + h) - f(θ - h)) / (2h)
    end
    abs(J) < 1e-14 && error("Degenerate PQ score derivative near θ=$θ")
    se = sqrt(mean(ψ .^ 2) / (J^2) / n)
    ψd = w .* 0 .+ J   # store constant J per obs for bootstrap scaling
    # better: store per-obs kernel contribution; constant is fine for se aggregation
    return θ, se, ψ, fill(J, n)
end

# ---- DoubleMLPQ model -------------------------------------------------------

"""
    DoubleMLPQ

Double machine learning for **potential quantiles** of binary treatment.

Estimates `θ_τ(d)` defined by `P(Y(d) ≤ θ_τ(d)) = τ` for `d ∈ {0,1}` using
a nonlinear Neyman-orthogonal score (nested cross-fitting).

# Example
```julia
data = make_irm_data(n_obs=1000, dim_x=5, theta=0.5; seed=1)
pq = DoubleMLPQ(
    data,
    LogisticRegressionLearner(α=0.5),
    LogisticRegressionLearner(α=0.5);
    treatment=1, quantile=0.5, n_folds=5,
)
fit!(pq)
summary_table(pq)
```
"""
mutable struct DoubleMLPQ <: AbstractDoubleML
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

function DoubleMLPQ(data::DoubleMLData, ml_g, ml_m;
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
        throw(ArgumentError("DoubleMLPQ requires binary treatment in {0,1}"))
    is_classifier(ml_g) || @warn "ml_g should be a classifier (predict_proba for CDF labels)"
    is_classifier(ml_m) || @warn "ml_m should be a classifier (propensity)"

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()

    n = n_obs(data)
    tname = "PQ(d=$treatment,τ=$(quantile))"
    return DoubleMLPQ(
        data, ml_g, ml_m, Int(treatment), Float64(quantile),
        n_folds, n_rep, Float64(trimming_threshold), normalize_ipw, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1),
        fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [tname],
        nothing, false, rng,
    )
end

function fit!(m::DoubleMLPQ; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.trimming_threshold

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
        g_hat, m_treat, ipw0, m_raw = _nuisance_est_pq(
            m.ml_g, m.ml_m, X, y, d, folds, m.treatment, m.quantile,
            ε, m.normalize_ipw, m.rng,
        )
        θ, se, ψ, ψd = _pq_estimate(y, d, g_hat, m_treat, m.treatment, m.quantile, ipw0)
        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = ψ
        psi_d_arr[:, r, 1] = ψd
        g_preds[:, r] = g_hat
        m_preds[:, r] = m_raw
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef
    m.se = se
    m.all_coef = all_coef
    m.all_se = all_se
    m.psi = psi_arr
    m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_g" => g_preds, "ml_m" => m_preds)
    end
    m.fitted = true
    return m
end
