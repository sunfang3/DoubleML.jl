"""
    DoubleMLPLR

Double machine learning for the **partially linear regression** model:

```
Y = D θ₀ + g₀(X) + ζ,    E[ζ | D, X] = 0
D = m₀(X) + V,           E[V | X] = 0
```

Supports multiple treatment columns (`data.d_mat`) and cluster-in-fit SE.

# Scores
- `"partialling out"` (default): residual-on-residual
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
    smpls_cluster::Union{Nothing,Vector}
    n_folds_per_cluster::Int
    var_scaling::Union{Nothing,Vector{Float64}}
    is_cluster_data::Bool
    cluster_dict::Union{Nothing,NamedTuple}
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

    n = n_obs(data)
    n_t = n_treat(data)
    is_cl = is_cluster_data(data)
    smpls, smpls_cluster, n_fpc = if draw_sample_splitting
        s, sc, nf = init_sample_splitting(data, n_folds, n_rep; rng=rng)
        s, sc, nf
    else
        Vector{Any}(), nothing, n_folds
    end

    return DoubleMLPLR(
        data, ml_l, ml_m, ml_g, n_folds, n_rep, String(score), smpls,
        smpls_cluster, n_fpc, nothing, is_cl, nothing,
        Float64[], Float64[],
        zeros(n_t, n_rep), zeros(n_t, n_rep),
        fill(NaN, n, n_rep, n_t),
        fill(NaN, n, n_rep, n_t),
        Dict{String,Matrix{Float64}}(),
        copy(data.d_cols),
        nothing, nothing, nothing,
        false, rng,
    )
end

function fit!(m::DoubleMLPLR; store_predictions::Bool=true,
              external_predictions=nothing)
    data = m.data
    y = data.y
    n = n_obs(data)
    n_rep = m.n_rep
    n_t = n_treat(data)
    is_cl = is_cluster_data(data)

    if isempty(m.smpls)
        smpls, smpls_cluster, n_fpc = init_sample_splitting(data, m.n_folds, n_rep; rng=m.rng)
        m.smpls = smpls
        m.smpls_cluster = smpls_cluster
        m.n_folds_per_cluster = n_fpc
        m.is_cluster_data = is_cl
    end

    all_coef = zeros(n_t, n_rep)
    all_se = zeros(n_t, n_rep)
    psi_arr = fill(NaN, n, n_rep, n_t)
    psi_d_arr = fill(NaN, n, n_rep, n_t)
    var_scaling = fill(Float64(n), n_t)

    # store predictions for first treatment only (compat)
    l_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)
    g_preds = fill(NaN, n, n_rep)

    sigma2_v = zeros(n_rep)
    nu2_v = zeros(n_rep)
    psi_s = fill(NaN, n, n_rep)
    psi_n = fill(NaN, n, n_rep)
    rr_m = fill(NaN, n, n_rep)

    for j in 1:n_t
        X, d, tname = design_for_treatment(data, j)
        d = collect(d)
        # per-treatment nested external predictions (Python style)
        ext_j = if external_predictions isa AbstractDict &&
                   (haskey(external_predictions, tname) || haskey(external_predictions, Symbol(tname)))
            haskey(external_predictions, tname) ? external_predictions[tname] :
                external_predictions[Symbol(tname)]
        else
            external_predictions
        end
        for r in 1:n_rep
            folds = m.smpls[r]
            use_clf_m = is_classifier(m.ml_m)

            ℓ̂ = something(_apply_external_pred(ext_j, "ml_l", r, n),
                           cross_fit_predict(m.ml_l, X, y, folds; classifier=false))
            m̂ = something(_apply_external_pred(ext_j, "ml_m", r, n),
                           cross_fit_predict(m.ml_m, X, d, folds; classifier=use_clf_m))

            if m.score == "IV-type"
                v = d .- m̂
                u = y .- ℓ̂
                θ0 = sum(v .* u) / sum(v .* v)
                ĝ = something(_apply_external_pred(ext_j, "ml_g", r, n),
                               cross_fit_predict(m.ml_g, X, y .- θ0 .* d, folds; classifier=false))
                psi_a = -(d .- m̂) .* d
                psi_b = (d .- m̂) .* (y .- ĝ)
                if j == 1
                    g_preds[:, r] = ĝ
                end
            else
                v = d .- m̂
                u = y .- ℓ̂
                psi_a = -(v .* v)
                psi_b = v .* u
            end

            θ = est_coef_linear(psi_a, psi_b)
            se, vsf = se_from_score(psi_a, psi_b, θ;
                                    smpls=folds,
                                    cluster=is_cl ? data.cluster : nothing,
                                    smpls_cluster=is_cl ? m.smpls_cluster[r] : nothing,
                                    n_folds_per_cluster=m.n_folds_per_cluster)

            all_coef[j, r] = θ
            all_se[j, r] = se
            psi_arr[:, r, j] = psi_a .* θ .+ psi_b
            psi_d_arr[:, r, j] = psi_a
            var_scaling[j] = vsf

            if j == 1
                l_preds[:, r] = ℓ̂
                m_preds[:, r] = m̂
                if m.score == "IV-type"
                    σ2, ν2, ps, pn, rr = sensitivity_elements_plr(y, d, g_preds[:, r], m̂, θ; score="IV-type")
                else
                    σ2, ν2, ps, pn, rr = sensitivity_elements_plr(y, d, ℓ̂, m̂, θ; score="partialling out")
                end
                sigma2_v[r] = σ2
                nu2_v[r] = ν2
                psi_s[:, r] = ps
                psi_n[:, r] = pn
                rr_m[:, r] = rr
            end
        end
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef
    m.se = se
    m.all_coef = all_coef
    m.all_se = all_se
    m.psi = psi_arr
    m.psi_deriv = psi_d_arr
    m.var_scaling = var_scaling
    m.treat_names = copy(data.d_cols)
    m.boot = nothing
    m.sens_elements = SensitivityElements(sigma2_v, nu2_v, psi_s, psi_n, rr_m)
    m.sensitivity = nothing
    if is_cl
        m.cluster_dict = (
            smpls = m.smpls,
            smpls_cluster = m.smpls_cluster,
            cluster_vars = data.cluster,
            n_folds_per_cluster = m.n_folds_per_cluster,
        )
    end
    if store_predictions
        m.predictions = Dict("ml_l" => l_preds, "ml_m" => m_preds)
        if m.score == "IV-type"
            m.predictions["ml_g"] = g_preds
        end
    end
    m.fitted = true
    return m
end
