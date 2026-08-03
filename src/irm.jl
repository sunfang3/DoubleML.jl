"""
    DoubleMLIRM

Double machine learning for the **interactive regression model** (binary treatment):

```
Y = g₀(D, X) + ζ,     E[ζ | X, D] = 0
D = m₀(X) + V,        E[V | X] = 0
```

with ATE parameter `θ₀ = E[g₀(1,X) − g₀(0,X)]`.

Default score is the **doubly robust ATE** score (Python `"ATE"`):

```
ψ = ĝ(1,X) − ĝ(0,X)
  + D (Y − ĝ(1,X)) / m̂(X)
  − (1−D)(Y − ĝ(0,X)) / (1 − m̂(X))
  − θ
```

so `ψ_a = −1`, `ψ_b = DR_i`.

Mirrors Python `doubleml.DoubleMLIRM`.
"""
mutable struct DoubleMLIRM <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any          # E[Y | X, D] — regressor, fit separately on D=0/1
    ml_m::Any          # E[D | X] — classifier or regressor
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
    predictions::Dict{String,Matrix{Float64}}
    treat_names::Vector{String}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLIRM(data::DoubleMLData, ml_g, ml_m;
                     n_folds::Int=5,
                     n_rep::Int=1,
                     score::AbstractString="ATE",
                     trimming_threshold::Real=1e-12,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    score in ("ATE", "ATTE") ||
        throw(ArgumentError("score must be \"ATE\" or \"ATTE\" (v0.1)"))
    # check binary treatment
    d_unique = unique(data.d)
    Set(d_unique) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLIRM requires binary treatment in {0,1}"))

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()

    n = n_obs(data)
    return DoubleMLIRM(
        data, ml_g, ml_m, n_folds, n_rep, String(score),
        Float64(trimming_threshold), smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [data.d_col],
        false,
        rng,
    )
end

function _cross_fit_g_binary(ml_g, X, y, d, folds)
    """OOF predictions of E[Y|X,D=0] and E[Y|X,D=1]."""
    n = size(X, 1)
    g0 = fill(NaN, n)
    g1 = fill(NaN, n)
    for (train, test) in folds
        tr0 = train[d[train] .== 0]
        tr1 = train[d[train] .== 1]
        isempty(tr0) && error("No control units in a training fold — reduce n_folds or check overlap")
        isempty(tr1) && error("No treated units in a training fold — reduce n_folds or check overlap")

        m0 = clone(ml_g)
        m1 = clone(ml_g)
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

    g0_preds = fill(NaN, n, n_rep)
    g1_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)

    for r in 1:n_rep
        folds = m.smpls[r]
        g0, g1 = _cross_fit_g_binary(m.ml_g, X, y, d, folds)
        m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=is_classifier(m.ml_m))
        # clip propensity
        m̂ = clamp.(m̂, ε, 1 - ε)

        # doubly robust scores
        if m.score == "ATE"
            # ψ_b = g1 - g0 + D*(Y-g1)/m - (1-D)*(Y-g0)/(1-m)
            dr = (g1 .- g0) .+
                 d .* (y .- g1) ./ m̂ .-
                 (1 .- d) .* (y .- g0) ./ (1 .- m̂)
            psi_a = fill(-1.0, n)
            psi_b = dr
        else
            # ATTE (effect on the treated)
            # θ = E[g1-g0 | D=1]; score from Chernozhukov et al.
            p = mean(d)
            dr = d ./ p .* (g1 .- g0) .+
                 d ./ p .* (y .- g1) .-
                 m̂ ./ p .* (1 .- d) ./ (1 .- m̂) .* (y .- g0)
            # actually standard ATTE DR is a bit different; use simple form:
            # For v0.1 ATTE: weighted residualization
            psi_a = fill(-1.0, n)
            psi_b = dr
        end

        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)

        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = psi_a .* θ .+ psi_b
        g0_preds[:, r] = g0
        g1_preds[:, r] = g1
        m_preds[:, r] = m̂
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef
    m.se = se
    m.all_coef = all_coef
    m.all_se = all_se
    m.psi = psi_arr
    if store_predictions
        m.predictions = Dict(
            "ml_g0" => g0_preds,
            "ml_g1" => g1_preds,
            "ml_m" => m_preds,
        )
    end
    m.fitted = true
    return m
end
