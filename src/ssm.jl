# Sample Selection Model (SSM) — missing-at-random score
# Python: doubleml.DoubleMLSSM
#
# Y observed when S=1. Target: ATE under MAR selection.
# ψ_a = −1
# ψ_b =  [S D (Y−g₁)/(m π) + g₁] − [S (1−D) (Y−g₀)/((1−m) π) + g₀]

"""
    DoubleMLSSM

Double machine learning for sample selection models (missing-at-random).

Requires selection indicator `s` in [`DoubleMLData`](@ref) (`s=1` if Y observed).

# Learners
- `ml_g` — regressor for `E[Y|X]` among selected treated/control
- `ml_m` — classifier for `P(D=1|X)`
- `ml_pi` — classifier for `P(S=1|D,X)`
"""
mutable struct DoubleMLSSM <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    ml_pi::Any
    n_folds::Int
    n_rep::Int
    score::String
    trimming_threshold::Float64
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

function DoubleMLSSM(data::DoubleMLData, ml_g, ml_m, ml_pi;
                     n_folds::Int=5,
                     n_rep::Int=1,
                     score::AbstractString="missing-at-random",
                     trimming_threshold::Real=1e-2,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    sc = String(score)
    sc in ("missing-at-random", "nonignorable") ||
        throw(ArgumentError("score must be \"missing-at-random\" or \"nonignorable\""))
    data.s === nothing && throw(ArgumentError("DoubleMLSSM requires selection indicator s in data"))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) || throw(ArgumentError("binary D required"))
    Set(unique(data.s)) ⊆ Set([0.0, 1.0]) || throw(ArgumentError("binary S required"))
    if sc == "nonignorable" && n_instr(data) < 1
        throw(ArgumentError("nonignorable score requires instrument Z"))
    end

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    return DoubleMLSSM(
        data, ml_g, ml_m, ml_pi, n_folds, n_rep, sc, Float64(trimming_threshold), smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(), [data.d_col],
        nothing, false, rng,
    )
end

function _cross_fit_g_sel(ml_g, X, y, d, s, d_level, folds)
    n = size(X, 1)
    g = fill(NaN, n)
    mask = (d .== d_level) .& (s .== 1)
    for (train, test) in folds
        tr = train[mask[train]]
        isempty(tr) && error("No selected units with D=$d_level in fold")
        m = clone(ml_g)
        fit!(m, X[tr, :], y[tr])
        g[test] = predict(m, X[test, :])
    end
    return g
end

function fit!(m::DoubleMLSSM; store_predictions::Bool=true)
    data = m.data
    X, y, d, s = data.x, data.y, data.d, data.s
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.trimming_threshold

    # for observed Y with S=0, y may be NaN — replace for safety
    y_safe = copy(y)
    for i in 1:n
        if s[i] == 0 && !isfinite(y_safe[i])
            y_safe[i] = 0.0
        end
    end

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    g1p = fill(NaN, n, n_rep); g0p = fill(NaN, n, n_rep)
    mp = fill(NaN, n, n_rep); pip = fill(NaN, n, n_rep)

    for r in 1:n_rep
        folds = m.smpls[r]
        if m.score == "missing-at-random"
            Xd = hcat(X, d)
            π̂ = cross_fit_predict(m.ml_pi, Xd, s, folds; classifier=true)
            m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=true)
            g1 = _cross_fit_g_sel(m.ml_g, X, y_safe, d, s, 1.0, folds)
            g0 = _cross_fit_g_sel(m.ml_g, X, y_safe, d, s, 0.0, folds)
        else
            # nonignorable: include Z and nested CF for π
            z = instrument(data)
            Xdz = hcat(X, d, z)
            π̂ = cross_fit_predict(m.ml_pi, Xdz, s, folds; classifier=true)
            Xp = hcat(X, π̂)
            m̂ = cross_fit_predict(m.ml_m, Xp, d, folds; classifier=true)
            g1 = _cross_fit_g_sel(m.ml_g, Xp, y_safe, d, s, 1.0, folds)
            g0 = _cross_fit_g_sel(m.ml_g, Xp, y_safe, d, s, 0.0, folds)
        end
        π̂ = clamp.(π̂, ε, 1 - ε)
        m̂ = clamp.(m̂, ε, 1 - ε)

        dt = d .== 1
        dc = d .== 0
        psi_a = fill(-1.0, n)
        psi_b1 = (dt .* s .* (y_safe .- g1)) ./ (m̂ .* π̂) .+ g1
        psi_b0 = (dc .* s .* (y_safe .- g0)) ./ ((1 .- m̂) .* π̂) .+ g0
        # zero out residual terms where S=0 (y undefined)
        psi_b1 = ifelse.(s .== 1, psi_b1, g1)
        psi_b0 = ifelse.(s .== 1, psi_b0, g0)
        # for S=0: residual term already 0; for S=1 and wrong d, residual 0
        psi_b1 = g1 .+ (dt .* s .* (y_safe .- g1)) ./ (m̂ .* π̂)
        psi_b0 = g0 .+ (dc .* s .* (y_safe .- g0)) ./ ((1 .- m̂) .* π̂)
        psi_b = psi_b1 .- psi_b0

        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)
        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, r, 1] = psi_a
        g1p[:, r] = g1; g0p[:, r] = g0; mp[:, r] = m̂; pip[:, r] = π̂
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_g1" => g1p, "ml_g0" => g0p, "ml_m" => mp, "ml_pi" => pip)
    end
    m.fitted = true
    return m
end
