"""
    DoubleMLPLR

Double machine learning for the **partially linear regression** model:

```
Y = D θ₀ + g₀(X) + ζ,    E[ζ | D, X] = 0
D = m₀(X) + V,           E[V | X] = 0
```

# Scores
- `"partialling out"` (default): residual-on-residual
  `ψ_a = -(D − m̂)²`, `ψ_b = (D − m̂)(Y − ℓ̂)`
- `"IV-type"`: uses an additional `ml_g` for `E[Y − Dθ | X]`

Mirrors Python `doubleml.DoubleMLPLR`.
"""
mutable struct DoubleMLPLR <: AbstractDoubleML
    data::DoubleMLData
    ml_l::Any
    ml_m::Any
    ml_g::Any
    n_folds::Int
    n_rep::Int
    score::String
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

function DoubleMLPLR(data::DoubleMLData, ml_l, ml_m;
                     ml_g=nothing,
                     n_folds::Int=5,
                     n_rep::Int=1,
                     score::AbstractString="partialling out",
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    score in ("partialling out", "IV-type") ||
        throw(ArgumentError("score must be \"partialling out\" or \"IV-type\""))
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))
    n_rep >= 1 || throw(ArgumentError("n_rep must be ≥ 1"))

    if score == "IV-type" && ml_g === nothing
        ml_g = clone(ml_l)
    end

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()

    n = n_obs(data)
    return DoubleMLPLR(
        data, ml_l, ml_m, ml_g, n_folds, n_rep, String(score), smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1),
        fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [data.d_col],
        nothing,
        false,
        rng,
    )
end

function fit!(m::DoubleMLPLR; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    n = n_obs(data)
    n_rep = m.n_rep

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)

    l_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)
    g_preds = fill(NaN, n, n_rep)

    for r in 1:n_rep
        folds = m.smpls[r]
        use_clf_m = is_classifier(m.ml_m)

        ℓ̂ = cross_fit_predict(m.ml_l, X, y, folds; classifier=false)
        m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=use_clf_m)

        if m.score == "IV-type"
            v = d .- m̂
            u = y .- ℓ̂
            θ0 = sum(v .* u) / sum(v .* v)
            ĝ = cross_fit_predict(m.ml_g, X, y .- θ0 .* d, folds; classifier=false)
            psi_a = -(d .- m̂) .* d
            psi_b = (d .- m̂) .* (y .- ĝ)
            g_preds[:, r] = ĝ
        else
            v = d .- m̂
            u = y .- ℓ̂
            psi_a = -(v .* v)
            psi_b = v .* u
        end

        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)

        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, r, 1] = psi_a
        l_preds[:, r] = ℓ̂
        m_preds[:, r] = m̂
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
        m.predictions = Dict("ml_l" => l_preds, "ml_m" => m_preds)
        if m.score == "IV-type"
            m.predictions["ml_g"] = g_preds
        end
    end
    m.fitted = true
    return m
end
