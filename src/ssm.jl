# Sample Selection Model (SSM)
# Python: doubleml.DoubleMLSSM
#
# Scores:
# - missing-at-random: π = P(S=1|D,X), m = P(D=1|X), g_d = E[Y|X,D=d,S=1]
# - nonignorable: nested CF with instrument Z:
#     train half1 → π = P(S=1|D,X,Z); half2 → m,g on (X, π̂)

"""
    DoubleMLSSM

Double machine learning for sample selection models.

Requires selection indicator `s` in [`DoubleMLData`](@ref) (`s=1` if Y observed).

# Scores
- `"missing-at-random"` — default MAR
- `"nonignorable"` — requires instrument `Z`; **nested cross-fitting** (Python parity)

# Learners
- `ml_g` — outcome among selected
- `ml_m` — treatment propensity
- `ml_pi` — selection propensity
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
    ps_processor::PSProcessor
    normalize_ipw::Bool
    # Python nonignorable nested CF uses train_test_split(..., random_state=42)
    nested_random_state::Int
    smpls::Vector
    coef::Vector{Float64}
    se::Vector{Float64}
    all_coef::Matrix{Float64}
    all_se::Matrix{Float64}
    psi::Array{Float64,3}
    psi_deriv::Array{Float64,3}
    predictions::Dict{String,Matrix{Float64}}
    models::Dict{String,Any}
    treat_names::Vector{String}
    boot::Union{Nothing,BootstrapResult}
    fitted::Bool
    rng::AbstractRNG
    ml_params::Dict{String,Any}
end

function DoubleMLSSM(data::DoubleMLData, ml_g, ml_m, ml_pi;
                     n_folds::Int=5,
                     n_rep::Int=1,
                     score::AbstractString="missing-at-random",
                     trimming_threshold::Real=1e-2,
                     ps_processor::Union{Nothing,PSProcessor}=nothing,
                     normalize_ipw::Bool=false,
                     nested_random_state::Integer=42,
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
    psp = resolve_ps_processor(ps_processor, trimming_threshold)
    return DoubleMLSSM(
        data, ml_g, ml_m, ml_pi, n_folds, n_rep, sc, Float64(trimming_threshold),
        psp, normalize_ipw, Int(nested_random_state), smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(), Dict{String,Any}(),
        [data.d_col], nothing, false, rng, Dict{String,Any}(),
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

"""
Stratified 50/50 split of `train` indices (Python
`train_test_split(..., test_size=0.5, stratify=..., random_state=nested_random_state)`).

# Notes
- Each call should use a **fresh** RNG seeded with the same `random_state` (Python
  reseeds every fold to 42).
- Per-stratum: shuffle then put first `floor(n/2)` into half1 (fit π), rest into half2
  (fit m/g). Rare strata with 1 obs go to half1 (π), matching sklearn's "need ≥2 for
  stratify" by falling back gracefully.
"""
function _stratified_half_split(train::AbstractVector{Int}, strata::AbstractVector,
                                rng::AbstractRNG)
    g1 = Int[]; g2 = Int[]
    # stable order of labels like sklearn's unique order of appearance after sort
    labs = sort(unique(strata[train]))
    for lab in labs
        idx = train[strata[train] .== lab]
        n = length(idx)
        if n == 0
            continue
        elseif n == 1
            # cannot stratify 50/50; put sole obs in half1 (π), leave half2 empty for this stratum
            push!(g1, idx[1])
            continue
        end
        idx = Random.shuffle(rng, collect(idx))
        # sklearn test_size=0.5 → n_test ≈ n/2 goes to "test" (half2), rest to train (half1)
        n_test = n ÷ 2                    # floor, matches int(n*0.5) for even/odd
        n_train = n - n_test
        # half1 = train_inds_1 (π), half2 = train_inds_2 (m,g)
        append!(g1, idx[1:n_train])
        append!(g2, idx[(n_train + 1):end])
    end
    # if half2 empty (all strata size 1), move half of g1 over
    if isempty(g2) && length(g1) >= 2
        mid = length(g1) ÷ 2
        g2 = g1[(mid + 1):end]
        g1 = g1[1:mid]
    end
    return sort(g1), sort(g2)
end

"""Nested CF for nonignorable SSM (Python DoubleMLSSM score='nonignorable')."""
function _ssm_nonignorable_nuisance(ml_g, ml_m, ml_pi, X, y, d, s, z, folds, ε;
                                    nested_random_state::Int=42,
                                    store_models::Bool=false)
    n = length(y)
    π̂ = fill(NaN, n)
    m̂ = fill(NaN, n)
    g1 = fill(NaN, n)
    g0 = fill(NaN, n)
    models = store_models ? Dict{String,Vector{Any}}(
        "ml_pi" => Any[], "ml_m" => Any[], "ml_g1" => Any[], "ml_g0" => Any[]
    ) : nothing

    dx = hcat(X, d, z)
    strata = Int.(d .+ 2 .* s)  # 0,1,2,3

    for (train, test) in folds
        # Python: train_test_split(..., random_state=42) — reseed every fold
        rng_nest = MersenneTwister(nested_random_state)
        tr1, tr2 = _stratified_half_split(train, strata, rng_nest)
        isempty(tr1) && error("Empty nested half1 for π — check strata balance")
        isempty(tr2) && error("Empty nested half2 for m/g — check strata balance")

        # half 1: fit π = P(S=1 | D,X,Z)
        mpi = clone(ml_pi)
        fit!(mpi, dx[tr1, :], s[tr1])
        # predict π on full sample for constructing xπ (as Python)
        π_full = predict_proba(mpi, dx)
        π_full = clamp.(π_full, ε, 1 - ε)
        π̂[test] = π_full[test]

        # half 2: m and g on (X, π)
        xpi = hcat(X, π_full)
        mm = clone(ml_m)
        fit!(mm, xpi[tr2, :], d[tr2])
        m̂[test] = clamp.(predict_proba(mm, xpi[test, :]), ε, 1 - ε)

        tr2_d1s1 = tr2[(d[tr2] .== 1) .& (s[tr2] .== 1)]
        tr2_d0s1 = tr2[(d[tr2] .== 0) .& (s[tr2] .== 1)]
        isempty(tr2_d1s1) && error("Empty D=1,S=1 in nested train half")
        isempty(tr2_d0s1) && error("Empty D=0,S=1 in nested train half")
        mg1 = clone(ml_g); fit!(mg1, xpi[tr2_d1s1, :], y[tr2_d1s1])
        mg0 = clone(ml_g); fit!(mg0, xpi[tr2_d0s1, :], y[tr2_d0s1])
        g1[test] = predict(mg1, xpi[test, :])
        g0[test] = predict(mg0, xpi[test, :])

        if store_models
            push!(models["ml_pi"], mpi)
            push!(models["ml_m"], mm)
            push!(models["ml_g1"], mg1)
            push!(models["ml_g0"], mg0)
        end
    end
    return π̂, m̂, g0, g1, models
end

function fit!(m::DoubleMLSSM; store_predictions::Bool=true, store_models::Bool=false)
    data = m.data
    X, y, d, s = data.x, data.y, data.d, data.s
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.ps_processor.clipping_threshold

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
    models_rep = Any[]

    for r in 1:n_rep
        folds = m.smpls[r]
        fold_models = nothing
        if m.score == "missing-at-random"
            Xd = hcat(X, d)
            π̂ = cross_fit_predict(m.ml_pi, Xd, s, folds; classifier=true)
            m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=true)
            g1 = _cross_fit_g_sel(m.ml_g, X, y_safe, d, s, 1.0, folds)
            g0 = _cross_fit_g_sel(m.ml_g, X, y_safe, d, s, 0.0, folds)
        else
            z = instrument(data)
            π̂, m̂, g0, g1, fold_models = _ssm_nonignorable_nuisance(
                m.ml_g, m.ml_m, m.ml_pi, X, y_safe, d, s, z, folds, ε;
                nested_random_state=m.nested_random_state,
                store_models=store_models,
            )
        end
        π̂ = process_propensity(π̂, m.ps_processor)
        m̂ = process_propensity(m̂, m.ps_processor)
        if m.normalize_ipw
            m̂ = _normalize_ipw(m̂, d)
            m̂ = process_propensity(m̂, m.ps_processor)
        end

        dt = d .== 1
        dc = d .== 0
        psi_a = fill(-1.0, n)
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
        store_models && push!(models_rep, fold_models)
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_g1" => g1p, "ml_g0" => g0p, "ml_m" => mp, "ml_pi" => pip)
    end
    m.models = store_models ? Dict{String,Any}("reps" => models_rep) : Dict{String,Any}()
    m.fitted = true
    return m
end
