# Regression Discontinuity (RDFlex-style residualization + local linear)
# Python: doubleml.rdd.RDFlex (simplified; no rdrobust dependency)

"""
    DoubleMLRDD

Sharp or fuzzy RDD with ML residualization of outcome (and treatment) near the cutoff,
then weighted local linear estimation (triangular kernel).

# Data
[`DoubleMLData`](@ref) with:
- `score` — running variable
- `y`, `d` (for sharp, `d = 1{score ≥ cutoff}` is fine; for fuzzy, actual treatment)
- `x` — covariates for residualization
"""
mutable struct DoubleMLRDD <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    cutoff::Float64
    fuzzy::Bool
    n_folds::Int
    n_rep::Int
    h::Float64                 # bandwidth (if NaN, rule-of-thumb)
    kernel::String
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
    h_used::Float64
end

function DoubleMLRDD(data::DoubleMLData, ml_g, ml_m=nothing;
                     cutoff::Real=0.0,
                     fuzzy::Bool=false,
                     n_folds::Int=5,
                     n_rep::Int=1,
                     h::Real=NaN,
                     kernel::AbstractString="triangular",
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    data.score === nothing && throw(ArgumentError("RDD requires running variable in data.score"))
    fuzzy && ml_m === nothing && throw(ArgumentError("fuzzy RDD requires ml_m"))
    kernel in ("triangular", "uniform", "epanechnikov") ||
        throw(ArgumentError("kernel must be triangular/uniform/epanechnikov"))

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    return DoubleMLRDD(
        data, ml_g, ml_m, Float64(cutoff), fuzzy, n_folds, n_rep, Float64(h),
        String(kernel), smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Any}(), ["LATE_RD"],
        nothing, false, rng, NaN,
    )
end

function _kernel_w(u::AbstractVector, kernel::String)
    a = abs.(u)
    if kernel == "triangular"
        return ifelse.(a .< 1, 1 .- a, 0.0)
    elseif kernel == "uniform"
        return ifelse.(a .< 1, 1.0, 0.0)
    else  # epanechnikov
        return ifelse.(a .< 1, 0.75 .* (1 .- a .^ 2), 0.0)
    end
end

function _rot_bandwidth(score::AbstractVector, cutoff::Real)
    s = score .- cutoff
    n = length(s)
    σ = std(s)
    # Silverman-like ROT for RD
    return 1.84 * σ * n^(-1 / 5)
end

"""Local linear weighted regression of y on [1, s, d, d*s] or sharp: treat = 1{s≥0}."""
function _local_linear_rd(s, y, d, w)
    n = length(y)
    # design: intercept, running, treatment, interaction
    X = hcat(ones(n), s, d, d .* s)
    # weighted LS: sqrt(w) X, sqrt(w) y
    sw = sqrt.(max.(w, 0.0))
    Xw = X .* sw
    yw = y .* sw
    # drop zero-weight rows
    keep = sw .> 0
    sum(keep) < 8 && error("Too few observations inside bandwidth for RDD")
    Xk = Xw[keep, :]
    yk = yw[keep]
    β = Xk \ yk
    # treatment effect is coefficient on d (at s=0)
    resid = y .- X * β
    # HC0 se for β[3]
    bread = inv(Xk' * Xk)
    meat = Xk' * (Xk .* (resid[keep] .^ 2))
    Ω = bread * meat * bread
    se = sqrt(max(Ω[3, 3], 0.0))
    return β[3], se, resid
end

function fit!(m::DoubleMLRDD; store_predictions::Bool=true)
    data = m.data
    X, y = data.x, data.y
    s_raw = data.score
    cutoff = m.cutoff
    s = s_raw .- cutoff
    n = n_obs(data)
    n_rep = m.n_rep

    h = isnan(m.h) ? _rot_bandwidth(s_raw, cutoff) : m.h
    m.h_used = h
    w0 = _kernel_w(s ./ h, m.kernel)

    # intended treatment (sharp assignment)
    z = Float64.(s .>= 0)
    d = m.fuzzy ? data.d : z

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)

    for r in 1:n_rep
        folds = m.smpls[r]
        # residualize Y: η_Y ≈ (g+ + g-)/2 estimated by ML on each side with kernel weights
        # practical: fit g on left and right with features X, average predictions
        left = s .< 0
        right = s .>= 0
        gL = fill(NaN, n); gR = fill(NaN, n)
        for (train, test) in folds
            trL = train[left[train] .& (w0[train] .> 0)]
            trR = train[right[train] .& (w0[train] .> 0)]
            if length(trL) >= 5
                mL = clone(m.ml_g); fit!(mL, X[trL, :], y[trL])
                gL[test] = predict(mL, X[test, :])
            end
            if length(trR) >= 5
                mR = clone(m.ml_g); fit!(mR, X[trR, :], y[trR])
                gR[test] = predict(mR, X[test, :])
            end
        end
        # fill missing side with other side
        for i in 1:n
            if !isfinite(gL[i]); gL[i] = isfinite(gR[i]) ? gR[i] : 0.0; end
            if !isfinite(gR[i]); gR[i] = isfinite(gL[i]) ? gL[i] : 0.0; end
        end
        ηY = 0.5 .* (gL .+ gR)
        My = y .- ηY

        if m.fuzzy
            mL = fill(NaN, n); mR = fill(NaN, n)
            for (train, test) in folds
                trL = train[left[train] .& (w0[train] .> 0)]
                trR = train[right[train] .& (w0[train] .> 0)]
                if length(trL) >= 5
                    mm = clone(m.ml_m); fit!(mm, X[trL, :], d[trL])
                    mL[test] = is_classifier(mm) ? predict_proba(mm, X[test, :]) : predict(mm, X[test, :])
                end
                if length(trR) >= 5
                    mm = clone(m.ml_m); fit!(mm, X[trR, :], d[trR])
                    mR[test] = is_classifier(mm) ? predict_proba(mm, X[test, :]) : predict(mm, X[test, :])
                end
            end
            for i in 1:n
                if !isfinite(mL[i]); mL[i] = isfinite(mR[i]) ? mR[i] : 0.0; end
                if !isfinite(mR[i]); mR[i] = isfinite(mL[i]) ? mL[i] : 0.0; end
            end
            ηD = 0.5 .* (mL .+ mR)
            Md = d .- ηD
            # fuzzy: local linear 2SLS — residualize, then IV of My on Md with instrument z
            # reduced form + first stage local linear
            rf, se_rf, _ = _local_linear_rd(s, My, z, w0)
            fs, se_fs, _ = _local_linear_rd(s, Md, z, w0)
            abs(fs) < 1e-8 && error("Weak first stage in fuzzy RDD")
            θ = rf / fs
            # delta method se
            se = abs(θ) * sqrt((se_rf / rf)^2 + (se_fs / fs)^2)
            resid = My .- θ .* Md
        else
            θ, se, resid = _local_linear_rd(s, My, z, w0)
        end

        all_coef[1, r] = θ
        all_se[1, r] = se
        # influence approx for bootstrap
        ψ = w0 .* resid
        psi_arr[:, r, 1] = ψ .- mean(ψ)
        psi_d_arr[:, r, 1] .= -1.0
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    m.fitted = true
    return m
end
