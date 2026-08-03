"""
    DoubleMLIRM

Double machine learning for the **interactive regression model** (binary treatment).

Default score is the **doubly robust ATE** score (Python `"ATE"`).
Mirrors Python `doubleml.DoubleMLIRM`.
"""
mutable struct DoubleMLIRM <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    n_folds::Int
    n_rep::Int
    score::String
    trimming_threshold::Float64
    weights::Union{Nothing,Vector{Float64}}
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
    sens_elements::Union{Nothing,SensitivityElements}
    sensitivity::Union{Nothing,SensitivityResult}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLIRM(data::DoubleMLData, ml_g, ml_m;
                     n_folds::Int=5,
                     n_rep::Int=1,
                     score::AbstractString="ATE",
                     trimming_threshold::Real=1e-12,
                     weights::Union{Nothing,AbstractVector}=nothing,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    score in ("ATE", "ATTE") ||
        throw(ArgumentError("score must be \"ATE\" or \"ATTE\""))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLIRM requires binary treatment in {0,1}"))
    n = n_obs(data)
    w = if weights === nothing
        nothing
    else
        length(weights) == n || throw(DimensionMismatch("weights length must equal n"))
        Float64.(weights)
    end

    smpls = draw_sample_splitting ?
        make_repeated_folds(n, n_folds, n_rep; rng=rng) :
        Vector{Any}()

    return DoubleMLIRM(
        data, ml_g, ml_m, n_folds, n_rep, String(score),
        Float64(trimming_threshold), w, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1),
        fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [data.d_col],
        nothing,
        nothing,
        nothing,
        false,
        rng,
    )
end

function _cross_fit_g_binary(ml_g, X, y, d, folds)
    n = size(X, 1)
    g0 = fill(NaN, n)
    g1 = fill(NaN, n)
    for (train, test) in folds
        tr0 = train[d[train] .== 0]
        tr1 = train[d[train] .== 1]
        isempty(tr0) && error("No control units in a training fold")
        isempty(tr1) && error("No treated units in a training fold")
        m0 = clone(ml_g); m1 = clone(ml_g)
        fit!(m0, X[tr0, :], y[tr0])
        fit!(m1, X[tr1, :], y[tr1])
        g0[test] = predict(m0, X[test, :])
        g1[test] = predict(m1, X[test, :])
    end
    return g0, g1
end

function fit!(m::DoubleMLIRM; store_predictions::Bool=true)
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

    g0_preds = fill(NaN, n, n_rep)
    g1_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)

    sigma2_v = zeros(n_rep)
    nu2_v = zeros(n_rep)
    psi_s = fill(NaN, n, n_rep)
    psi_n = fill(NaN, n, n_rep)
    rr_m = fill(NaN, n, n_rep)

    for r in 1:n_rep
        folds = m.smpls[r]
        g0, g1 = _cross_fit_g_binary(m.ml_g, X, y, d, folds)
        m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=is_classifier(m.ml_m))
        m̂ = clamp.(m̂, ε, 1 - ε)

        if m.score == "ATE"
            dr = (g1 .- g0) .+
                 d .* (y .- g1) ./ m̂ .-
                 (1 .- d) .* (y .- g0) ./ (1 .- m̂)
            psi_a = fill(-1.0, n)
            psi_b = dr
        else
            p = mean(d)
            dr = d ./ p .* (g1 .- g0) .+
                 d ./ p .* (y .- g1) .-
                 m̂ ./ p .* (1 .- d) ./ (1 .- m̂) .* (y .- g0)
            psi_a = fill(-1.0, n)
            psi_b = dr
        end
        # optional observation weights (normalized to mean 1)
        if m.weights !== nothing
            w = m.weights ./ mean(m.weights)
            psi_a = psi_a .* w
            psi_b = psi_b .* w
        end

        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)
        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, r, 1] = psi_a
        g0_preds[:, r] = g0
        g1_preds[:, r] = g1
        m_preds[:, r] = m̂

        # sensitivity (ATE formula; used for ATTE as approximation)
        σ2, ν2, ps, pn, rr = sensitivity_elements_irm_ate(y, d, g0, g1, m̂)
        sigma2_v[r] = σ2
        nu2_v[r] = ν2
        psi_s[:, r] = ps
        psi_n[:, r] = pn
        rr_m[:, r] = rr
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    m.sens_elements = SensitivityElements(sigma2_v, nu2_v, psi_s, psi_n, rr_m)
    m.sensitivity = nothing
    if store_predictions
        m.predictions = Dict("ml_g0" => g0_preds, "ml_g1" => g1_preds, "ml_m" => m_preds)
    end
    m.fitted = true
    return m
end
