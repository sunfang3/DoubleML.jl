"""
    DoubleMLIIVM

Double machine learning for the **interactive IV model** (binary treatment & binary instrument).

```
Y = g₀(Z, X) + ν,     E[ν | Z, X] = 0
D = r₀(Z, X) + U,     E[U | Z, X] = 0
Z = m₀(X) + V,        E[V | X] = 0
```

Target parameter is the **local average treatment effect (LATE)**:

```
θ₀ = (E[g₀(1,X)] − E[g₀(0,X)]) / (E[r₀(1,X)] − E[r₀(0,X)])
```

# Learners
- `ml_g` — outcome regression ``E[Y | X]`` fit separately on ``Z=0`` and ``Z=1``
- `ml_m` — instrument propensity ``E[Z | X]`` (classifier preferred)
- `ml_r` — treatment propensity ``E[D | X]`` fit separately on ``Z=0`` and ``Z=1``

# Score
`"LATE"` (default) — doubly robust score of Chernozhukov et al. / Python `DoubleMLIIVM`.

Mirrors Python `doubleml.DoubleMLIIVM`.
"""
mutable struct DoubleMLIIVM <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    ml_r::Any
    n_folds::Int
    n_rep::Int
    score::String
    trimming_threshold::Float64
    always_takers::Bool
    never_takers::Bool
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

function DoubleMLIIVM(data::DoubleMLData, ml_g, ml_m, ml_r;
                      n_folds::Int=5,
                      n_rep::Int=1,
                      score::AbstractString="LATE",
                      trimming_threshold::Real=1e-2,
                      always_takers::Bool=true,
                      never_takers::Bool=true,
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    score == "LATE" || throw(ArgumentError("score must be \"LATE\""))
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))
    n_rep >= 1 || throw(ArgumentError("n_rep must be ≥ 1"))

    # binary treatment
    Set(unique(data.d)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLIIVM requires binary treatment D ∈ {0,1}"))

    # single binary instrument
    data.z === nothing && throw(ArgumentError(
        "DoubleMLIIVM requires a binary instrument (set z / z_cols in DoubleMLData)"))
    n_instr(data) == 1 || throw(ArgumentError(
        "DoubleMLIIVM requires exactly one binary instrument"))
    z = vec(data.z)
    Set(unique(z)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLIIVM requires binary instrument Z ∈ {0,1}"))

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()

    n = n_obs(data)
    return DoubleMLIIVM(
        data, ml_g, ml_m, ml_r, n_folds, n_rep, String(score),
        Float64(trimming_threshold), always_takers, never_takers, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [data.d_col],
        false,
        rng,
    )
end

"""
Out-of-fold predictions of E[Y|X] (or E[D|X]) stratified by a binary condition `cond`
on the *training* fold only (test predictions still made for all test rows).
"""
function _cross_fit_conditional(ml, X, y, cond::AbstractVector{Bool}, folds;
                                classifier::Bool=false)
    n = size(X, 1)
    preds = fill(NaN, n)
    for (train, test) in folds
        tr = train[cond[train]]
        isempty(tr) && error(
            "Empty conditional training fold — reduce n_folds or check support of Z")
        m = clone(ml)
        fit!(m, X[tr, :], y[tr])
        if classifier || is_classifier(m)
            preds[test] = predict_proba(m, X[test, :])
        else
            preds[test] = predict(m, X[test, :])
        end
    end
    return preds
end

function fit!(m::DoubleMLIIVM; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    z = vec(data.z)
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.trimming_threshold

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)

    g0p = fill(NaN, n, n_rep)
    g1p = fill(NaN, n, n_rep)
    mp = fill(NaN, n, n_rep)
    r0p = fill(NaN, n, n_rep)
    r1p = fill(NaN, n, n_rep)

    z0 = z .== 0
    z1 = z .== 1

    for rep in 1:n_rep
        folds = m.smpls[rep]

        # g0 = E[Y|X,Z=0], g1 = E[Y|X,Z=1]
        g0 = _cross_fit_conditional(m.ml_g, X, y, z0, folds; classifier=false)
        g1 = _cross_fit_conditional(m.ml_g, X, y, z1, folds; classifier=false)

        # m = E[Z|X]
        m̂ = cross_fit_predict(m.ml_m, X, z, folds; classifier=is_classifier(m.ml_m))
        m̂ = clamp.(m̂, ε, 1 - ε)

        # r0 = E[D|X,Z=0], r1 = E[D|X,Z=1]
        if m.always_takers
            r0 = _cross_fit_conditional(m.ml_r, X, d, z0, folds;
                                        classifier=is_classifier(m.ml_r))
            r0 = clamp.(r0, 0.0, 1.0)
        else
            r0 = zeros(n)
        end
        if m.never_takers
            r1 = _cross_fit_conditional(m.ml_r, X, d, z1, folds;
                                        classifier=is_classifier(m.ml_r))
            r1 = clamp.(r1, 0.0, 1.0)
        else
            r1 = ones(n)
        end

        # LATE score
        # ψ_b = g1 − g0 + Z(Y−g1)/m − (1−Z)(Y−g0)/(1−m)
        # ψ_a = −[ r1 − r0 + Z(D−r1)/m − (1−Z)(D−r0)/(1−m) ]
        u0 = y .- g0
        u1 = y .- g1
        w0 = d .- r0
        w1 = d .- r1

        psi_b = (g1 .- g0) .+
                z .* u1 ./ m̂ .-
                (1 .- z) .* u0 ./ (1 .- m̂)
        psi_a = -((r1 .- r0) .+
                  z .* w1 ./ m̂ .-
                  (1 .- z) .* w0 ./ (1 .- m̂))

        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)

        all_coef[1, rep] = θ
        all_se[1, rep] = se
        psi_arr[:, rep, 1] = psi_a .* θ .+ psi_b
        g0p[:, rep] = g0
        g1p[:, rep] = g1
        mp[:, rep] = m̂
        r0p[:, rep] = r0
        r1p[:, rep] = r1
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef
    m.se = se
    m.all_coef = all_coef
    m.all_se = all_se
    m.psi = psi_arr
    if store_predictions
        m.predictions = Dict(
            "ml_g0" => g0p,
            "ml_g1" => g1p,
            "ml_m" => mp,
            "ml_r0" => r0p,
            "ml_r1" => r1p,
        )
    end
    m.fitted = true
    return m
end
