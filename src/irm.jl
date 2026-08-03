"""
    DoubleMLIRM

Double machine learning for the **interactive regression model** (binary treatment).

Default score is the **doubly robust ATE** score (Python `"ATE"`).
Supports multiple binary treatment columns and cluster-in-fit SE.
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
    ml_params::Dict{String,Any}
    models::Dict{String,Any}
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
    for j in 1:n_treat(data)
        dj = @view data.d_mat[:, j]
        Set(unique(dj)) ⊆ Set([0.0, 1.0]) ||
            throw(ArgumentError("DoubleMLIRM requires binary treatment in {0,1} (column $(data.d_cols[j]))"))
    end
    n = n_obs(data)
    n_t = n_treat(data)
    w = if weights === nothing
        nothing
    else
        length(weights) == n || throw(DimensionMismatch("weights length must equal n"))
        Float64.(weights)
    end

    is_cl = is_cluster_data(data)
    smpls, smpls_cluster, n_fpc = if draw_sample_splitting
        init_sample_splitting(data, n_folds, n_rep; rng=rng)
    else
        Vector{Any}(), nothing, n_folds
    end

    return DoubleMLIRM(
        data, ml_g, ml_m, n_folds, n_rep, String(score),
        Float64(trimming_threshold), w, smpls,
        smpls_cluster, n_fpc, nothing, is_cl, nothing,
        Float64[], Float64[],
        zeros(n_t, n_rep), zeros(n_t, n_rep),
        fill(NaN, n, n_rep, n_t),
        fill(NaN, n, n_rep, n_t),
        Dict{String,Matrix{Float64}}(),
        copy(data.d_cols),
        nothing, nothing, nothing,
        false, rng,
        Dict{String,Any}(), Dict{String,Any}(),
    )
end

function _cross_fit_g_binary(ml_g, X, y, d, folds; store_models::Bool=false,
                             params_factory=nothing)
    n = size(X, 1)
    g0 = fill(NaN, n)
    g1 = fill(NaN, n)
    models0 = store_models ? Any[] : nothing
    models1 = store_models ? Any[] : nothing
    for (k, (train, test)) in enumerate(folds)
        tr0 = train[d[train] .== 0]
        tr1 = train[d[train] .== 1]
        isempty(tr0) && error("No control units in a training fold")
        isempty(tr1) && error("No treated units in a training fold")
        p = params_factory === nothing ? nothing : params_factory(k)
        m0 = p === nothing ? clone(ml_g) : _clone_with_params(ml_g, p)
        m1 = p === nothing ? clone(ml_g) : _clone_with_params(ml_g, p)
        fit!(m0, X[tr0, :], y[tr0])
        fit!(m1, X[tr1, :], y[tr1])
        g0[test] = predict(m0, X[test, :])
        g1[test] = predict(m1, X[test, :])
        if store_models
            push!(models0, m0); push!(models1, m1)
        end
    end
    return g0, g1, models0, models1
end

function fit!(m::DoubleMLIRM; store_predictions::Bool=true,
              store_models::Bool=false,
              external_predictions=nothing)
    data = m.data
    y = data.y
    n = n_obs(data)
    n_rep = m.n_rep
    n_t = n_treat(data)
    ε = m.trimming_threshold
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

    g0_preds = fill(NaN, n, n_rep)
    g1_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)

    sigma2_v = zeros(n_rep)
    nu2_v = zeros(n_rep)
    psi_s = fill(NaN, n, n_rep)
    psi_n = fill(NaN, n, n_rep)
    rr_m = fill(NaN, n, n_rep)
    models_store = Dict{String,Any}()

    for j in 1:n_t
        X, d, tname = design_for_treatment(data, j)
        d = collect(d)
        ml_g = _learner_with_params(m, m.ml_g, "ml_g", tname)
        ml_m = _learner_with_params(m, m.ml_m, "ml_m", tname)
        ext_j = if external_predictions isa AbstractDict &&
                   (haskey(external_predictions, tname) || haskey(external_predictions, Symbol(tname)))
            haskey(external_predictions, tname) ? external_predictions[tname] :
                external_predictions[Symbol(tname)]
        else
            external_predictions
        end
        for r in 1:n_rep
            folds = m.smpls[r]
            g_pf = _fold_params_factory(m, "ml_g", tname, r)
            m_pf = _fold_params_factory(m, "ml_m", tname, r)
            g0_ext = _apply_external_pred(ext_j, "ml_g0", r, n)
            g1_ext = _apply_external_pred(ext_j, "ml_g1", r, n)
            local models0, models1
            if g0_ext === nothing || g1_ext === nothing
                g0, g1, models0, models1 = _cross_fit_g_binary(
                    ml_g, X, y, d, folds; store_models=store_models, params_factory=g_pf)
                g0_ext !== nothing && (g0 = g0_ext)
                g1_ext !== nothing && (g1 = g1_ext)
            else
                g0, g1 = g0_ext, g1_ext
                models0 = models1 = nothing
            end
            m̂, m_models = if (extm = _apply_external_pred(ext_j, "ml_m", r, n)) !== nothing
                extm, nothing
            else
                cross_fit_predict_store(ml_m, X, d, folds;
                                       classifier=is_classifier(ml_m),
                                       params_factory=m_pf,
                                       store_models=store_models)
            end
            m̂ = clamp.(m̂, ε, 1 - ε)
            if store_models && j == 1 && r == 1
                models_store["ml_g0"] = models0
                models_store["ml_g1"] = models1
                models_store["ml_m"] = m_models
            end

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
            if m.weights !== nothing
                w = m.weights ./ mean(m.weights)
                psi_a = psi_a .* w
                psi_b = psi_b .* w
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
                g0_preds[:, r] = g0
                g1_preds[:, r] = g1
                m_preds[:, r] = m̂
                σ2, ν2, ps, pn, rr = sensitivity_elements_irm_ate(y, d, g0, g1, m̂)
                sigma2_v[r] = σ2
                nu2_v[r] = ν2
                psi_s[:, r] = ps
                psi_n[:, r] = pn
                rr_m[:, r] = rr
            end
        end
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
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
        m.predictions = Dict("ml_g0" => g0_preds, "ml_g1" => g1_preds, "ml_m" => m_preds)
    end
    m.models = store_models ? models_store : Dict{String,Any}()
    m.fitted = true
    return m
end
