# Average Potential Outcomes (APO / APOS)
# Python: doubleml.DoubleMLAPO / DoubleMLAPOS
#
# θ_j = E[g_0(d_j, X)] with score
#   ψ_a = −w/mean(w),  ψ_b = w·ĝ_j + w̄·1{D=d_j}(Y−ĝ_j)/m_j

"""
    DoubleMLAPO

Average potential outcome at a single treatment level `treatment_level`:

```
θ = E[Y(d)]   for  d = treatment_level
```

Uses the doubly robust APO score (unit weights by default).
"""
mutable struct DoubleMLAPO <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    treatment_level::Float64
    n_folds::Int
    n_rep::Int
    trimming_threshold::Float64
    normalize_ipw::Bool
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
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLAPO(data::DoubleMLData, ml_g, ml_m;
                     treatment_level::Real=1,
                     n_folds::Int=5,
                     n_rep::Int=1,
                     trimming_threshold::Real=1e-2,
                     normalize_ipw::Bool=false,
                     weights::Union{Nothing,AbstractVector}=nothing,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    tl = Float64(treatment_level)
    any(isapprox.(data.d, tl; atol=1e-12)) ||
        throw(ArgumentError("treatment_level=$tl not present in data.d"))
    if weights !== nothing
        length(weights) == n_obs(data) || throw(DimensionMismatch("weights length"))
    end

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    return DoubleMLAPO(
        data, ml_g, ml_m, tl, n_folds, n_rep, Float64(trimming_threshold),
        normalize_ipw,
        weights === nothing ? nothing : Float64.(weights),
        smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        ["APO(d=$tl)"],
        nothing, false, rng,
    )
end

"""Cross-fit E[Y|X, D=level] and predict for all rows."""
function _cross_fit_g_level(ml_g, X, y, d, level, folds)
    n = size(X, 1)
    g = fill(NaN, n)
    treated = isapprox.(d, level; atol=1e-12)
    for (train, test) in folds
        tr = train[treated[train]]
        isempty(tr) && error("No units with D=$level in a training fold")
        m = clone(ml_g)
        fit!(m, X[tr, :], y[tr])
        g[test] = predict(m, X[test, :])
    end
    return g
end

function fit!(m::DoubleMLAPO; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    n = n_obs(data)
    n_rep = m.n_rep
    ε = m.trimming_threshold
    tl = m.treatment_level
    w = m.weights === nothing ? ones(n) : m.weights
    w_mean = mean(w)
    abs(w_mean) < 1e-14 && error("mean(weights) ≈ 0")

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    g_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)

    treated = Float64.(isapprox.(d, tl; atol=1e-12))

    for r in 1:n_rep
        folds = m.smpls[r]
        ĝ = _cross_fit_g_level(m.ml_g, X, y, d, tl, folds)
        # propensity for 1{D = level}
        m̂ = cross_fit_predict(m.ml_m, X, treated, folds; classifier=is_classifier(m.ml_m))
        m̂ = clamp.(m̂, ε, 1 - ε)
        if m.normalize_ipw
            m̂ = _normalize_ipw(m̂, treated)
            m̂ = clamp.(m̂, ε, 1 - ε)
        end

        u = y .- ĝ
        psi_a = -w ./ w_mean
        psi_b = w .* ĝ .+ w .* treated .* u ./ m̂

        θ = est_coef_linear(psi_a, psi_b)
        se = se_linear(psi_a, psi_b, θ)
        all_coef[1, r] = θ
        all_se[1, r] = se
        psi_arr[:, r, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, r, 1] = psi_a
        g_preds[:, r] = ĝ
        m_preds[:, r] = m̂
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_g" => g_preds, "ml_m" => m_preds)
    end
    m.fitted = true
    return m
end

# ---- APOS: multiple levels --------------------------------------------------

"""
    DoubleMLAPOS

Average potential outcomes for multiple discrete treatment levels, plus
[`causal_contrast`](@ref) for pairwise ATEs between levels.
"""
mutable struct DoubleMLAPOS <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    treatment_levels::Vector{Float64}
    n_folds::Int
    n_rep::Int
    trimming_threshold::Float64
    normalize_ipw::Bool
    weights::Union{Nothing,Vector{Float64}}
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
    modellist::Vector{DoubleMLAPO}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLAPOS(data::DoubleMLData, ml_g, ml_m,
                      treatment_levels;
                      n_folds::Int=5,
                      n_rep::Int=1,
                      trimming_threshold::Real=1e-2,
                      normalize_ipw::Bool=false,
                      weights::Union{Nothing,AbstractVector}=nothing,
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    levels = Float64.(collect(treatment_levels))
    length(levels) >= 1 || throw(ArgumentError("treatment_levels must be non-empty"))
    for lv in levels
        any(isapprox.(data.d, lv; atol=1e-12)) ||
            throw(ArgumentError("treatment level $lv not in data.d"))
    end

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    n_l = length(levels)
    names = ["APO(d=$lv)" for lv in levels]
    return DoubleMLAPOS(
        data, ml_g, ml_m, levels, n_folds, n_rep, Float64(trimming_threshold),
        normalize_ipw,
        weights === nothing ? nothing : Float64.(weights),
        smpls,
        Float64[], Float64[],
        zeros(n_l, n_rep), zeros(n_l, n_rep),
        fill(NaN, n, n_rep, n_l), fill(NaN, n, n_rep, n_l),
        Dict{String,Any}(), names,
        nothing, DoubleMLAPO[], false, rng,
    )
end

function fit!(m::DoubleMLAPOS; store_predictions::Bool=true)
    n = n_obs(m.data)
    n_rep = m.n_rep
    n_l = length(m.treatment_levels)
    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    m.modellist = DoubleMLAPO[]
    all_coef = zeros(n_l, n_rep)
    all_se = zeros(n_l, n_rep)
    psi_arr = fill(NaN, n, n_rep, n_l)
    psi_d_arr = fill(NaN, n, n_rep, n_l)

    for (j, lv) in enumerate(m.treatment_levels)
        apo = DoubleMLAPO(
            m.data, clone(m.ml_g), clone(m.ml_m);
            treatment_level=lv,
            n_folds=m.n_folds, n_rep=n_rep,
            trimming_threshold=m.trimming_threshold,
            normalize_ipw=m.normalize_ipw,
            weights=m.weights,
            draw_sample_splitting=false,
            rng=copy(m.rng),
        )
        apo.smpls = m.smpls
        fit!(apo; store_predictions=store_predictions)
        push!(m.modellist, apo)
        all_coef[j, :] = apo.all_coef[1, :]
        all_se[j, :] = apo.all_se[1, :]
        psi_arr[:, :, j] = apo.psi[:, :, 1]
        psi_d_arr[:, :, j] = apo.psi_deriv[:, :, 1]
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    m.fitted = true
    return m
end

"""
    causal_contrast(m::DoubleMLAPOS, reference_levels) -> DataFrame

Estimate causal contrasts `APO(d) − APO(ref)` for each treatment level vs
`reference_levels` (scalar or collection).
"""
function causal_contrast(m::DoubleMLAPOS, reference_levels)
    m.fitted || error("Call fit! first")
    refs = Float64.(reference_levels isa Number ? [reference_levels] : collect(reference_levels))
    n = n_obs(m.data)
    n_rep = m.n_rep
    contrasts = String[]
    coefs = Float64[]
    ses = Float64[]
    for ref in refs
        jref = findfirst(lv -> isapprox(lv, ref; atol=1e-12), m.treatment_levels)
        jref === nothing && throw(ArgumentError("reference level $ref not in treatment_levels"))
        for (j, lv) in enumerate(m.treatment_levels)
            isapprox(lv, ref; atol=1e-12) && continue
            all_diff = zeros(1, n_rep)
            all_se = zeros(1, n_rep)
            for r in 1:n_rep
                all_diff[1, r] = m.all_coef[j, r] - m.all_coef[jref, r]
                Jj = mean(@view m.psi_deriv[:, r, j])
                Jr = mean(@view m.psi_deriv[:, r, jref])
                IF = @view(m.psi[:, r, j]) ./ Jj .- @view(m.psi[:, r, jref]) ./ Jr
                all_se[1, r] = sqrt(mean(IF .^ 2) / n)
            end
            c, s = aggregate_reps(all_diff, all_se)
            push!(contrasts, "$lv vs $ref")
            push!(coefs, c[1])
            push!(ses, s[1])
        end
    end
    tstat = coefs ./ ses
    p = 2 .* cdf.(Normal(), -abs.(tstat))
    z = quantile(Normal(), 0.975)
    return DataFrame(
        contrast = contrasts,
        coef = coefs,
        std_err = ses,
        t = tstat,
        pvalue = p,
        ci_lower = coefs .- z .* ses,
        ci_upper = coefs .+ z .* ses,
    )
end
