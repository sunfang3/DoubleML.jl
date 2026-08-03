# Local Potential Quantiles (LPQ) — nonlinear DML under IV
# Python: doubleml.DoubleMLLPQ
#
# Identifies quantiles of Y(d) for compliers using binary instrument Z.
# Score (sign = 2d−1):
#   ψ(θ) = sign · ( g1−g0 + Z/m_z (1{D=d}1{Y≤θ}−g1)
#                 − (1−Z)/(1−m_z) (1{D=d}1{Y≤θ}−g0) ) / π_c  − τ
# where π_c is the complier probability.

"""
    DoubleMLLPQ

Double machine learning for **local potential quantiles** (compliers) with a
binary instrument `Z` stored in [`DoubleMLData`](@ref).

Both `ml_g` and `ml_m` should be **classifiers**.

# Example
```julia
data = make_iivm_data(n_obs=2000, dim_x=5, theta=0.5; seed=1)
lpq = DoubleMLLPQ(
    data,
    LogisticRegressionLearner(α=0.5),
    LogisticRegressionLearner(α=0.5);
    treatment=1, quantile=0.5, n_folds=5,
)
fit!(lpq)
summary_table(lpq)
```
"""
mutable struct DoubleMLLPQ <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    treatment::Int
    quantile::Float64
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
    predictions::Dict{String,Matrix{Float64}}
    treat_names::Vector{String}
    boot::Union{Nothing,BootstrapResult}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLLPQ(data::DoubleMLData, ml_g, ml_m;
                     treatment::Integer=1,
                     quantile::Real=0.5,
                     n_folds::Int=5,
                     n_rep::Int=1,
                     trimming_threshold::Real=1e-2,
                     ps_processor::Union{Nothing,PSProcessor}=nothing,
                     normalize_ipw::Bool=true,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    treatment in (0, 1) || throw(ArgumentError("treatment must be 0 or 1"))
    (0 < quantile < 1) || throw(ArgumentError("quantile must be in (0,1)"))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLLPQ requires binary treatment in {0,1}"))
    n_instr(data) >= 1 || throw(ArgumentError("DoubleMLLPQ requires an instrument Z in DoubleMLData"))
    z = instrument(data)  # single instrument
    Set(unique(z)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLLPQ requires binary instrument in {0,1}"))

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    tname = "LPQ(d=$treatment,τ=$(quantile))"
    return DoubleMLLPQ(
        data, ml_g, ml_m, Int(treatment), Float64(quantile),
        n_folds, n_rep, Float64(trimming_threshold),
        resolve_ps_processor(ps_processor, trimming_threshold),
        normalize_ipw, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(), [tname],
        nothing, false, rng,
    )
end

"""Stratified half-split using integer strata codes."""
function _stratified_half_codes(idx::AbstractVector{Int}, strata::AbstractVector, rng::AbstractRNG)
    # strata values can be 0,1,2,3 for d+2z
    idx_s = Random.shuffle(rng, collect(idx))
    # try to balance by putting alternating strata
    levels = unique(strata[idx_s])
    a = Int[]; b = Int[]
    for lev in levels
        il = idx_s[strata[idx_s] .== lev]
        mid = max(length(il) ÷ 2, 0)
        append!(a, il[1:mid])
        append!(b, il[mid+1:end])
    end
    if isempty(a) || isempty(b)
        mid = max(length(idx_s) ÷ 2, 1)
        a = sort(idx_s[1:mid]); b = sort(idx_s[mid+1:end])
    else
        a = sort(a); b = sort(b)
    end
    return a, b
end

function _nuisance_est_lpq(ml_g, ml_m, X, y, d, z, folds, treatment::Int, τ::Float64,
                           trimming::Float64, normalize_ipw::Bool, rng::AbstractRNG)
    n = length(y)
    n_folds = length(folds)
    m_z_hat = fill(NaN, n)
    m_d_z0_hat = fill(NaN, n)
    m_d_z1_hat = fill(NaN, n)
    g_du_z0_hat = fill(NaN, n)
    g_du_z1_hat = fill(NaN, n)
    ipw_vec = fill(NaN, n_folds)

    y_treat = y[d .== treatment]
    isempty(y_treat) && error("No observations with D=$treatment")
    coef_start = quantile(y_treat, τ)
    coef_bounds = (minimum(y), maximum(y))
    strata = d .+ 2 .* z   # 0..3

    for (i_fold, (train, test)) in enumerate(folds)
        tr1, tr2 = _stratified_half_codes(train, strata, rng)

        # --- prelim on tr1 ---
        d1, y1, z1 = d[tr1], y[tr1], z[tr1]
        n_inner = min(n_folds, max(2, length(tr1) ÷ 5))
        # m_z via inner CV
        m_z_prelim = _cv_predict_proba_on(ml_m, X, z, tr1, n_inner, rng)
        m_z_prelim = _clip_ps(m_z_prelim, trimming)
        if normalize_ipw
            m_z_prelim = _normalize_ipw(m_z_prelim, z1)
        end

        # m_d | z=0 and m_d | z=1 on tr1 (fit, predict on all tr1)
        z0_1 = z1 .== 0
        z1_1 = z1 .== 1
        (sum(z0_1) < 2 || sum(z1_1) < 2) && error("Too few Z cells in nested train fold")
        md0 = clone(ml_m); fit!(md0, X[tr1[z0_1], :], d1[z0_1])
        md1 = clone(ml_m); fit!(md1, X[tr1[z1_1], :], d1[z1_1])
        m_d_z0_p = _clip_ps(predict_proba(md0, X[tr1, :]), trimming)
        m_d_z1_p = _clip_ps(predict_proba(md1, X[tr1, :]), trimming)

        comp_prob_prelim = mean(
            m_d_z1_p .- m_d_z0_p .+
            z1 ./ m_z_prelim .* (d1 .- m_d_z1_p) .-
            (1 .- z1) ./ (1 .- m_z_prelim) .* (d1 .- m_d_z0_p)
        )
        abs(comp_prob_prelim) < 1e-8 && error("Degenerate preliminary complier probability")

        sign = 2 * treatment - 1.0
        function ipw_score(θ)
            w = sign .* (z1 ./ m_z_prelim .- (1 .- z1) ./ (1 .- m_z_prelim)) ./ comp_prob_prelim
            u = (d1 .== treatment) .* (y1 .<= θ)
            return mean(w .* u .- τ)
        end
        ok, bracket = _get_bracket_guess(ipw_score, coef_start, coef_bounds)
        ipw_est = ok ? _brent_root(ipw_score, bracket[1], bracket[2]) :
                      _minimize_abs(ipw_score, coef_bounds[1], coef_bounds[2])
        ipw_vec[i_fold] = ipw_est

        # --- g_du on tr2 by Z ---
        d2, y2, z2 = d[tr2], y[tr2], z[tr2]
        z0_2 = z2 .== 0
        z1_2 = z2 .== 1
        sum(z0_2) < 2 && error("Too few Z=0 in train2")
        sum(z1_2) < 2 && error("Too few Z=1 in train2")
        du0 = (d2[z0_2] .== treatment) .* (y2[z0_2] .<= ipw_est)
        du1 = (d2[z1_2] .== treatment) .* (y2[z1_2] .<= ipw_est)
        g0 = clone(ml_g); g1 = clone(ml_g)
        # constant label edge cases
        if length(unique(Float64.(du0))) < 2
            g_du_z0_hat[test] .= mean(Float64.(du0))
        else
            fit!(g0, X[tr2[z0_2], :], Float64.(du0))
            g_du_z0_hat[test] = predict_proba(g0, X[test, :])
        end
        if length(unique(Float64.(du1))) < 2
            g_du_z1_hat[test] .= mean(Float64.(du1))
        else
            fit!(g1, X[tr2[z1_2], :], Float64.(du1))
            g_du_z1_hat[test] = predict_proba(g1, X[test, :])
        end

        # --- refit propensities on full train ---
        mz = clone(ml_m); fit!(mz, X[train, :], z[train])
        m_z_hat[test] = predict_proba(mz, X[test, :])

        z_tr = z[train]
        md0f = clone(ml_m); fit!(md0f, X[train[z_tr .== 0], :], d[train[z_tr .== 0]])
        md1f = clone(ml_m); fit!(md1f, X[train[z_tr .== 1], :], d[train[z_tr .== 1]])
        m_d_z0_hat[test] = predict_proba(md0f, X[test, :])
        m_d_z1_hat[test] = predict_proba(md1f, X[test, :])
    end

    m_z_hat = _clip_ps(m_z_hat, trimming)
    if normalize_ipw
        m_z_hat = _normalize_ipw(m_z_hat, z)
    end
    m_d_z0_hat = _clip_ps(m_d_z0_hat, trimming)
    m_d_z1_hat = _clip_ps(m_d_z1_hat, trimming)
    g_du_z0_hat = clamp.(g_du_z0_hat, 0.0, 1.0)
    g_du_z1_hat = clamp.(g_du_z1_hat, 0.0, 1.0)

    comp_prob = mean(
        m_d_z1_hat .- m_d_z0_hat .+
        z ./ m_z_hat .* (d .- m_d_z1_hat) .-
        (1 .- z) ./ (1 .- m_z_hat) .* (d .- m_d_z0_hat)
    )
    abs(comp_prob) < 1e-8 && error("Degenerate complier probability")

    return m_z_hat, g_du_z0_hat, g_du_z1_hat, comp_prob, mean(ipw_vec)
end

function _lpq_score_mean(θ, y, d, z, m_z, g0, g1, comp_prob, treatment::Int, τ::Float64)
    sign = 2 * treatment - 1.0
    ind = d .== treatment
    s1 = g1 .- g0
    s2 = (z ./ m_z) .* (ind .* (y .<= θ) .- g1)
    s3 = ((1 .- z) ./ (1 .- m_z)) .* (ind .* (y .<= θ) .- g0)
    return mean(sign .* (s1 .+ s2 .- s3) ./ comp_prob .- τ)
end

function _lpq_score_vec(θ, y, d, z, m_z, g0, g1, comp_prob, treatment::Int, τ::Float64)
    sign = 2 * treatment - 1.0
    ind = d .== treatment
    s1 = g1 .- g0
    s2 = (z ./ m_z) .* (ind .* (y .<= θ) .- g1)
    s3 = ((1 .- z) ./ (1 .- m_z)) .* (ind .* (y .<= θ) .- g0)
    return sign .* (s1 .+ s2 .- s3) ./ comp_prob .- τ
end

function fit!(m::DoubleMLLPQ; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    z = instrument(data)
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.ps_processor.clipping_threshold
    τ = m.quantile

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    pred_mz = fill(NaN, n, n_rep)
    pred_g0 = fill(NaN, n, n_rep)
    pred_g1 = fill(NaN, n, n_rep)

    bounds = (minimum(y), maximum(y))

    for r in 1:n_rep
        folds = m.smpls[r]
        m_z, g0, g1, comp_prob, ipw0 = _nuisance_est_lpq(
            m.ml_g, m.ml_m, X, y, d, z, folds, m.treatment, τ,
            ε, m.normalize_ipw, m.rng,
        )
        f = θ -> _lpq_score_mean(θ, y, d, z, m_z, g0, g1, comp_prob, m.treatment, τ)
        ok, bracket = _get_bracket_guess(f, ipw0, bounds)
        θ = ok ? _brent_root(f, bracket[1], bracket[2]) :
                 _minimize_abs(f, bounds[1], bounds[2])

        ψ = _lpq_score_vec(θ, y, d, z, m_z, g0, g1, comp_prob, m.treatment, τ)
        sign = 2 * m.treatment - 1.0
        w = sign .* ((z ./ m_z) .- (1 .- z) ./ (1 .- m_z)) .* (d .== m.treatment) ./ comp_prob
        J = _weighted_kde_deriv(y .- θ, w)
        if abs(J) < 1e-10
            h = max(1e-4 * (bounds[2] - bounds[1]), 1e-6)
            J = (f(θ + h) - f(θ - h)) / (2h)
        end
        abs(J) < 1e-14 && error("Degenerate LPQ score derivative")
        se = sqrt(mean(ψ .^ 2) / (J^2) / n)

        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = ψ
        psi_d_arr[:, r, 1] .= J
        pred_mz[:, r] = m_z
        pred_g0[:, r] = g0
        pred_g1[:, r] = g1
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_m_z" => pred_mz, "ml_g_du_z0" => pred_g0, "ml_g_du_z1" => pred_g1)
    end
    m.fitted = true
    return m
end
