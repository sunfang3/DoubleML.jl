"""
    DoubleMLPLIV

Double machine learning for the **partially linear IV** model:

```
Y − D θ₀ = g₀(X) + ζ,    E[ζ | Z, X] = 0
Z = m₀(X) + V,           E[V | X] = 0
```

with policy variable `D` and instrument(s) `Z`.

# Learners (partial-X default, matching Python `DoubleMLPLIV`)
- `ml_l` — ``E[Y | X]``
- `ml_m` — ``E[Z | X]`` (one instrument) or per-instrument when ``n_instr > 1``
- `ml_r` — ``E[D | X]``
- `ml_g` — ``E[Y − Dθ | X]`` only for score `"IV-type"`

# Scores
- `"partialling out"` (default):
  ``ψ_a = −(D − r̂)(Z − m̂)``, ``ψ_b = (Z − m̂)(Y − ℓ̂)``
- `"IV-type"` (single instrument only):
  ``ψ_a = −(Z − m̂) D``, ``ψ_b = (Z − m̂)(Y − ĝ)``

Mirrors Python `doubleml.DoubleMLPLIV` (partial-X path).
"""
mutable struct DoubleMLPLIV <: AbstractDoubleML
    data::DoubleMLData
    ml_l::Any
    ml_m::Any
    ml_r::Any
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
    predictions::Dict{String,Matrix{Float64}}
    treat_names::Vector{String}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLPLIV(data::DoubleMLData, ml_l, ml_m, ml_r;
                      ml_g=nothing,
                      n_folds::Int=5,
                      n_rep::Int=1,
                      score::AbstractString="partialling out",
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    n_instr(data) > 0 || throw(ArgumentError(
        "DoubleMLPLIV requires at least one instrument (set z / z_cols in DoubleMLData). " *
        "Use DoubleMLPLR for models without instruments."))
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))
    n_rep >= 1 || throw(ArgumentError("n_rep must be ≥ 1"))

    if score == "IV-type"
        n_instr(data) == 1 || throw(ArgumentError(
            "score = \"IV-type\" is only supported for a single instrument"))
        if ml_g === nothing
            throw(ArgumentError("For score = \"IV-type\", ml_g must be specified"))
        end
    elseif score != "partialling out"
        throw(ArgumentError("score must be \"partialling out\" or \"IV-type\""))
    end

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()

    n = n_obs(data)
    return DoubleMLPLIV(
        data, ml_l, ml_m, ml_r, ml_g, n_folds, n_rep, String(score), smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [data.d_col],
        false,
        rng,
    )
end

"""Cross-fit multi-output predictions of E[Z_j | X] for each instrument column."""
function _cross_fit_predict_multi(ml, X::AbstractMatrix, Z::AbstractMatrix, folds)
    n, k = size(Z)
    preds = fill(NaN, n, k)
    for j in 1:k
        preds[:, j] = cross_fit_predict(ml, X, Z[:, j], folds; classifier=false)
    end
    return preds
end

"""Project residualized D on residualized multi-Z (2SLS first stage projection)."""
function _project_onto(V::AbstractMatrix, w::AbstractVector)
    # w_hat_tilde = V * β with intercept via demeaning
    n = size(V, 1)
    V1 = hcat(ones(n), V)
    β = V1 \ w
    return V1 * β
end

function fit!(m::DoubleMLPLIV; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    Z = data.z
    Z === nothing && error("instruments missing")
    n = n_obs(data)
    n_rep = m.n_rep
    k = size(Z, 2)

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)

    l_preds = fill(NaN, n, n_rep)
    # store instrument residuals product side: for k=1 store m_hat; for k>1 store projection
    m_preds = fill(NaN, n, n_rep)
    r_preds = fill(NaN, n, n_rep)
    g_preds = fill(NaN, n, n_rep)

    for rep in 1:n_rep
        folds = m.smpls[rep]

        ℓ̂ = cross_fit_predict(m.ml_l, X, y, folds; classifier=false)
        r̂ = cross_fit_predict(m.ml_r, X, d, folds; classifier=false)

        if k == 1
            z = vec(Z)
            m̂ = cross_fit_predict(m.ml_m, X, z, folds; classifier=false)
            u = y .- ℓ̂
            w = d .- r̂
            v = z .- m̂

            if m.score == "IV-type"
                # initial θ from partialling out
                θ0 = est_coef_linear(-(w .* v), v .* u)
                ĝ = cross_fit_predict(m.ml_g, X, y .- θ0 .* d, folds; classifier=false)
                psi_a = -(v .* d)
                psi_b = v .* (y .- ĝ)
                g_preds[:, rep] = ĝ
            else
                psi_a = -(w .* v)
                psi_b = v .* u
            end
            m_preds[:, rep] = m̂
        else
            # multi-instrument: partialling out via projection of w on V
            m.score == "partialling out" || error("multi-instrument requires partialling out")
            M = _cross_fit_predict_multi(m.ml_m, X, Z, folds)
            V = Z .- M
            w = d .- r̂
            u = y .- ℓ̂
            r_tilde = _project_onto(V, w)
            psi_a = -(w .* r_tilde)
            psi_b = r_tilde .* u
            m_preds[:, rep] = r_tilde  # store projected instrument residual for inspection
        end

        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)

        all_coef[1, rep] = θ
        all_se[1, rep] = se
        psi_arr[:, rep, 1] = psi_a .* θ .+ psi_b
        l_preds[:, rep] = ℓ̂
        r_preds[:, rep] = r̂
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef
    m.se = se
    m.all_coef = all_coef
    m.all_se = all_se
    m.psi = psi_arr
    if store_predictions
        m.predictions = Dict(
            "ml_l" => l_preds,
            "ml_m" => m_preds,
            "ml_r" => r_preds,
        )
        if m.score == "IV-type"
            m.predictions["ml_g"] = g_preds
        end
    end
    m.fitted = true
    return m
end
