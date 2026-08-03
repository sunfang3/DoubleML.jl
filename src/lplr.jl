# Logistic Partially Linear Regression (LPLR)
# E[Y|D,X] = expit(β D + r₀(X)), Y binary
# Python: doubleml.DoubleMLLPLR (Liu–Zhang–Zhou 2021)
#
# Practical implementation of the instrument-type score:
#   ψ(β) = (Y − expit(β D + r̂(X))) (D − m̂(X))
# with r̂ obtained via residualized logit of M̂ = P(Y=1|D,X).

_expit(z) = 1 / (1 + exp(-clamp(z, -50.0, 50.0)))
_logit(p) = log(clamp(p, 1e-8, 1 - 1e-8) / (1 - clamp(p, 1e-8, 1 - 1e-8)))

"""
    DoubleMLLPLR

Logistic partially linear model for binary outcomes:

```
P(Y=1 | D, X) = expit(β₀ D + r₀(X))
```

# Learners
- `ml_M` — classifier for `P(Y=1|D,X)` (features `[D X]`)
- `ml_t` — regressor for `E[logit(M)|X]`
- `ml_m` — regressor/classifier for `E[D|X]`
"""
mutable struct DoubleMLLPLR <: AbstractDoubleML
    data::DoubleMLData
    ml_M::Any
    ml_t::Any
    ml_m::Any
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

function DoubleMLLPLR(data::DoubleMLData, ml_M, ml_t, ml_m;
                      n_folds::Int=5,
                      n_rep::Int=1,
                      score::AbstractString="instrument",
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    sc = String(score)
    sc in ("instrument", "nuisance_space") ||
        throw(ArgumentError("score must be \"instrument\" or \"nuisance_space\""))
    Set(unique(data.y)) ⊆ Set([0.0, 1.0]) ||
        throw(ArgumentError("DoubleMLLPLR requires binary outcome Y ∈ {0,1}"))

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    return DoubleMLLPLR(
        data, ml_M, ml_t, ml_m, n_folds, n_rep, sc, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(), [data.d_col],
        nothing, false, rng,
    )
end

function fit!(m::DoubleMLLPLR; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    n = n_obs(data)
    n_rep = m.n_rep
    Xd = hcat(d, X)  # features for M

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    M_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)
    r_preds = fill(NaN, n, n_rep)

    for rep in 1:n_rep
        folds = m.smpls[rep]
        # M̂ = P(Y=1 | D, X)
        M̂ = cross_fit_predict(m.ml_M, Xd, y, folds; classifier=true)
        M̂ = clamp.(M̂, 1e-6, 1 - 1e-6)
        # m̂ = E[D|X]  (nuisance_space: train on Y=0 only)
        if m.score == "nuisance_space"
            m̂ = fill(NaN, n)
            for (train, test) in folds
                tr = train[y[train] .== 0]
                isempty(tr) && (tr = train)
                mm = clone(m.ml_m)
                fit!(mm, X[tr, :], d[tr])
                m̂[test] = is_classifier(mm) ? predict_proba(mm, X[test, :]) : predict(mm, X[test, :])
            end
        else
            m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=is_classifier(m.ml_m))
        end
        # â = E[D|X] full sample (same as m for continuous D)
        â = cross_fit_predict(m.ml_m, X, d, folds; classifier=is_classifier(m.ml_m))
        # preliminary β and residualization for r
        W = _logit.(M̂)
        # residualize W and D on X
        t̂ = cross_fit_predict(m.ml_t, X, W, folds; classifier=false)
        d_til = d .- â
        # pilot β
        den = sum(d_til .^ 2)
        abs(den) < 1e-12 && error("Degenerate residualized treatment in LPLR")
        β0 = sum(d_til .* (W .- t̂)) / den
        r̂ = t̂ .- β0 .* â

        # nonlinear root for instrument score
        f = β -> mean((y .- _expit.(β .* d .+ r̂)) .* d_til)
        # derivative for Newton / SE
        f′ = β -> begin
            μ = _expit.(β .* d .+ r̂)
            mean(-d .* μ .* (1 .- μ) .* d_til)
        end
        β = Float64(β0)
        for _ in 1:40
            g = f(β)
            gp = f′(β)
            abs(gp) < 1e-14 && break
            step = g / gp
            β -= step
            abs(step) < 1e-10 && break
        end
        # fallback bracket if needed
        if !isfinite(β) || abs(f(β)) > 1e-3
            lo, hi = β0 - 5, β0 + 5
            ok, br = _get_bracket_guess(f, β0, (lo, hi))
            β = ok ? _brent_root(f, br[1], br[2]) : _minimize_abs(f, lo, hi)
        end

        ψ = (y .- _expit.(β .* d .+ r̂)) .* d_til
        J = f′(β)
        abs(J) < 1e-14 && error("Degenerate LPLR score derivative")
        se = sqrt(mean(ψ .^ 2) / (J^2) / n)

        all_coef[1, rep] = β
        all_se[1, rep] = se
        psi_arr[:, rep, 1] = ψ
        psi_d_arr[:, rep, 1] .= J
        M_preds[:, rep] = M̂
        m_preds[:, rep] = m̂
        r_preds[:, rep] = r̂
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_M" => M_preds, "ml_m" => m_preds, "r_hat" => r_preds)
    end
    m.fitted = true
    return m
end
