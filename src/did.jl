# Two-period Difference-in-Differences (ATT)
# Aligned with Python DoubleMLDID / Sant'Anna & Zhao (2020) scores.
#
# Outcome `y` should be the change Y₁ − Y₀ (or a single post-period outcome
# under appropriate conditioning). Treatment `d` is binary group indicator.

"""
    DoubleMLDID

Two-period difference-in-differences ATT via double machine learning.

# Scores
- `"observational"` (default): conditional parallel trends — needs `ml_m`
- `"experimental"`: treatment independent of covariates (A/B style)

# Example
```julia
# y = Y_post - Y_pre, d = treatment group
did = DoubleMLDID(data, ml_g, ml_m; score="observational")
fit!(did)
summary_table(did)
```
"""
mutable struct DoubleMLDID <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    n_folds::Int
    n_rep::Int
    score::String
    in_sample_normalization::Bool
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

function DoubleMLDID(data::DoubleMLData, ml_g, ml_m=nothing;
                     n_folds::Int=5,
                     n_rep::Int=1,
                     score::AbstractString="observational",
                     in_sample_normalization::Bool=true,
                     trimming_threshold::Real=1e-2,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    sc = String(score)
    sc in ("observational", "experimental") ||
        throw(ArgumentError("score must be \"observational\" or \"experimental\""))
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLDID requires binary treatment in {0,1}"))
    if sc == "observational"
        ml_m === nothing && throw(ArgumentError("ml_m required for score=\"observational\""))
    end

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    return DoubleMLDID(
        data, ml_g, ml_m, n_folds, n_rep, sc, in_sample_normalization,
        Float64(trimming_threshold), smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [data.d_col],
        nothing, false, rng,
    )
end

function _did_score_elements(y, d, g0, g1, m_hat, score::String, in_sample::Bool)
    n = length(y)
    p_hat = mean(d)
    resid0 = y .- g0

    if score == "observational"
        if in_sample
            weight_psi_a = d ./ mean(d)
            prop_w = (1 .- d) .* (m_hat ./ (1 .- m_hat))
            weight_resid = d ./ mean(d) .- prop_w ./ mean(prop_w)
        else
            weight_psi_a = d ./ p_hat
            weight_resid = (d .- m_hat) ./ (p_hat .* (1 .- m_hat))
        end
        psi_b1 = zeros(n)
    else
        # experimental
        if in_sample
            weight_psi_a = ones(n)
            weight_g0 = d ./ mean(d) .- 1
            weight_g1 = 1 .- d ./ mean(d)
            weight_resid = d ./ mean(d) .- (1 .- d) ./ mean(1 .- d)
        else
            weight_psi_a = ones(n)
            weight_g0 = d ./ p_hat .- 1
            weight_g1 = 1 .- d ./ p_hat
            weight_resid = (d .- p_hat) ./ (p_hat .* (1 .- p_hat))
        end
        psi_b1 = weight_g0 .* g0 .+ weight_g1 .* g1
    end
    psi_a = -weight_psi_a
    psi_b = psi_b1 .+ weight_resid .* resid0
    return psi_a, psi_b
end

function fit!(m::DoubleMLDID; store_predictions::Bool=true)
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

    for r in 1:n_rep
        folds = m.smpls[r]
        g0, g1 = _cross_fit_g_binary(m.ml_g, X, y, d, folds)
        if m.score == "observational"
            m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=is_classifier(m.ml_m))
            m̂ = clamp.(m̂, ε, 1 - ε)
        else
            m̂ = fill(mean(d), n)
        end
        psi_a, psi_b = _did_score_elements(y, d, g0, g1, m̂, m.score, m.in_sample_normalization)
        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)
        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, r, 1] = psi_a
        g0_preds[:, r] = g0
        g1_preds[:, r] = g1
        m_preds[:, r] = m̂
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_g0" => g0_preds, "ml_g1" => g1_preds, "ml_m" => m_preds)
    end
    m.fitted = true
    return m
end
