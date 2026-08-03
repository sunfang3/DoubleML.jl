"""
    DoubleMLPLIV

Double machine learning for the **partially linear IV** model.

# Partialling modes (Python classmethods)
| Mode | Constructor | Learners | Score |
|------|-------------|----------|-------|
| `:partialX` (default) | `DoubleMLPLIV(data, ml_l, ml_m, ml_r)` | ℓ=E[Y\\|X], m=E[Z\\|X], r=E[D\\|X] | partialling out / IV-type |
| `:partialZ` | `DoubleMLPLIV_partialZ(data, ml_r)` | r=E[D\\|X,Z] | partialling out |
| `:partialXZ` | `DoubleMLPLIV_partialXZ(data, ml_l, ml_m, ml_r)` | ℓ=E[Y\\|X], m=E[D\\|X,Z], r≈E[m\\|X] | partialling out |

Mirrors Python `doubleml.DoubleMLPLIV`.
"""
mutable struct DoubleMLPLIV <: AbstractDoubleML
    data::DoubleMLData
    ml_l::Any
    ml_m::Any
    ml_r::Any
    ml_g::Any
    partial_mode::Symbol   # :partialX | :partialZ | :partialXZ
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
    fitted::Bool
    rng::AbstractRNG
end

function _pliv_init(data, ml_l, ml_m, ml_r, ml_g, partial_mode, n_folds, n_rep, score,
                    draw_sample_splitting, rng)
    n_instr(data) > 0 || throw(ArgumentError(
        "DoubleMLPLIV requires at least one instrument (set z / z_cols)."))
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))
    n_rep >= 1 || throw(ArgumentError("n_rep must be ≥ 1"))
    n_treat(data) == 1 || throw(ArgumentError("DoubleMLPLIV supports a single treatment in this version"))
    partial_mode in (:partialX, :partialZ, :partialXZ) ||
        throw(ArgumentError("partial_mode must be :partialX, :partialZ, or :partialXZ"))

    if score == "IV-type"
        partial_mode == :partialX || throw(ArgumentError("IV-type only with :partialX"))
        n_instr(data) == 1 || throw(ArgumentError("IV-type requires a single instrument"))
        ml_g === nothing && throw(ArgumentError("IV-type requires ml_g"))
    elseif score != "partialling out"
        throw(ArgumentError("score must be \"partialling out\" or \"IV-type\""))
    end

    n = n_obs(data)
    is_cl = is_cluster_data(data)
    smpls, smpls_cluster, n_fpc = if draw_sample_splitting
        init_sample_splitting(data, n_folds, n_rep; rng=rng)
    else
        Vector{Any}(), nothing, n_folds
    end

    return DoubleMLPLIV(
        data, ml_l, ml_m, ml_r, ml_g, partial_mode, n_folds, n_rep, String(score), smpls,
        smpls_cluster, n_fpc, nothing, is_cl, nothing,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Matrix{Float64}}(),
        [data.d_col], nothing, false, rng,
    )
end

function _pliv_ensure_smpls!(m::DoubleMLPLIV)
    if isempty(m.smpls)
        smpls, smpls_cluster, n_fpc = init_sample_splitting(m.data, m.n_folds, m.n_rep; rng=m.rng)
        m.smpls = smpls
        m.smpls_cluster = smpls_cluster
        m.n_folds_per_cluster = n_fpc
        m.is_cluster_data = is_cluster_data(m.data)
    end
end

function _pliv_se(m::DoubleMLPLIV, psi_a, psi_b, θ, folds, rep)
    is_cl = is_cluster_data(m.data)
    return se_from_score(psi_a, psi_b, θ;
                         smpls=folds,
                         cluster=is_cl ? m.data.cluster : nothing,
                         smpls_cluster=is_cl ? m.smpls_cluster[rep] : nothing,
                         n_folds_per_cluster=m.n_folds_per_cluster)
end

"""Default: partial out X (Python `DoubleMLPLIV`)."""
function DoubleMLPLIV(data::DoubleMLData, ml_l, ml_m, ml_r;
                      ml_g=nothing,
                      n_folds::Int=5,
                      n_rep::Int=1,
                      score::AbstractString="partialling out",
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    return _pliv_init(data, ml_l, ml_m, ml_r, ml_g, :partialX, n_folds, n_rep, score,
                      draw_sample_splitting, rng)
end

"""
    DoubleMLPLIV_partialZ(data, ml_r; kwargs...)

Partial out instruments only: fit ``E[D | X, Z]`` and use
``ψ_a = −r̂ D``, ``ψ_b = r̂ Y`` (Python `DoubleMLPLIV._partialZ`).
"""
function DoubleMLPLIV_partialZ(data::DoubleMLData, ml_r;
                               n_folds::Int=5,
                               n_rep::Int=1,
                               draw_sample_splitting::Bool=true,
                               rng::AbstractRNG=Random.default_rng())
    return _pliv_init(data, nothing, nothing, ml_r, nothing, :partialZ, n_folds, n_rep,
                      "partialling out", draw_sample_splitting, rng)
end

"""
    DoubleMLPLIV_partialXZ(data, ml_l, ml_m, ml_r; kwargs...)

Partial out both X and Z (Python `DoubleMLPLIV._partialXZ`).
"""
function DoubleMLPLIV_partialXZ(data::DoubleMLData, ml_l, ml_m, ml_r;
                                n_folds::Int=5,
                                n_rep::Int=1,
                                draw_sample_splitting::Bool=true,
                                rng::AbstractRNG=Random.default_rng())
    return _pliv_init(data, ml_l, ml_m, ml_r, nothing, :partialXZ, n_folds, n_rep,
                      "partialling out", draw_sample_splitting, rng)
end

function _cross_fit_predict_multi(ml, X::AbstractMatrix, Z::AbstractMatrix, folds)
    n, k = size(Z)
    preds = fill(NaN, n, k)
    for j in 1:k
        preds[:, j] = cross_fit_predict(ml, X, Z[:, j], folds; classifier=false)
    end
    return preds
end

function _project_onto(V::AbstractMatrix, w::AbstractVector)
    n = size(V, 1)
    V1 = hcat(ones(n), V)
    return V1 * (V1 \ w)
end

"""
Cross-fit E[D|X,Z] and return (oof_preds, train_pred_list) where
`train_pred_list[k]` is length-n NaN-filled with predictions on fold k's train set.
"""
function _cross_fit_with_train_preds(ml, X::AbstractMatrix, y::AbstractVector, folds)
    n = size(X, 1)
    oof = fill(NaN, n)
    train_list = Vector{Vector{Float64}}(undef, length(folds))
    for (k, (train, test)) in enumerate(folds)
        m = clone(ml)
        fit!(m, X[train, :], y[train])
        oof[test] = predict(m, X[test, :])
        tp = fill(NaN, n)
        tp[train] = predict(m, X[train, :])
        train_list[k] = tp
    end
    return oof, train_list
end

"""Cross-fit with fold-specific targets (list of length-n vectors, NaN outside train)."""
function _cross_fit_fold_targets(ml, X::AbstractMatrix, y_list::Vector{Vector{Float64}}, folds)
    n = size(X, 1)
    length(y_list) == length(folds) || error("y_list length must match n_folds")
    oof = fill(NaN, n)
    for (k, (train, test)) in enumerate(folds)
        yk = y_list[k]
        m = clone(ml)
        fit!(m, X[train, :], yk[train])
        oof[test] = predict(m, X[test, :])
    end
    return oof
end

function fit!(m::DoubleMLPLIV; store_predictions::Bool=true)
    if m.partial_mode == :partialX
        return _fit_pliv_partial_x!(m; store_predictions)
    elseif m.partial_mode == :partialZ
        return _fit_pliv_partial_z!(m; store_predictions)
    else
        return _fit_pliv_partial_xz!(m; store_predictions)
    end
end

function _fit_pliv_partial_x!(m::DoubleMLPLIV; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    Z = data.z
    n = n_obs(data)
    n_rep = m.n_rep
    k = size(Z, 2)
    _pliv_ensure_smpls!(m)

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    l_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)
    r_preds = fill(NaN, n, n_rep)
    g_preds = fill(NaN, n, n_rep)
    var_scaling = fill(Float64(n), 1)

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
            M = _cross_fit_predict_multi(m.ml_m, X, Z, folds)
            V = Z .- M
            w = d .- r̂
            u = y .- ℓ̂
            r_tilde = _project_onto(V, w)
            psi_a = -(w .* r_tilde)
            psi_b = r_tilde .* u
            m_preds[:, rep] = r_tilde
        end

        θ = est_coef_linear(psi_a, psi_b)
        se, vsf = _pliv_se(m, psi_a, psi_b, θ, folds, rep)
        all_coef[1, rep] = θ
        all_se[1, rep] = se
        var_scaling[1] = vsf
        psi_arr[:, rep, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, rep, 1] = psi_a
        l_preds[:, rep] = ℓ̂
        r_preds[:, rep] = r̂
    end

    _finalize_pliv!(m, all_coef, all_se, psi_arr, psi_d_arr, store_predictions,
                    Dict("ml_l" => l_preds, "ml_m" => m_preds, "ml_r" => r_preds,
                         "ml_g" => g_preds); var_scaling=var_scaling)
end

function _fit_pliv_partial_z!(m::DoubleMLPLIV; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    Z = data.z
    XZ = hcat(X, Z)
    n = n_obs(data)
    n_rep = m.n_rep
    _pliv_ensure_smpls!(m)

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    r_preds = fill(NaN, n, n_rep)
    var_scaling = fill(Float64(n), 1)

    for rep in 1:n_rep
        folds = m.smpls[rep]
        r̂ = cross_fit_predict(m.ml_r, XZ, d, folds; classifier=false)
        psi_a = -(r̂ .* d)
        psi_b = r̂ .* y
        θ = est_coef_linear(psi_a, psi_b)
        se, vsf = _pliv_se(m, psi_a, psi_b, θ, folds, rep)
        all_coef[1, rep] = θ
        all_se[1, rep] = se
        var_scaling[1] = vsf
        psi_arr[:, rep, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, rep, 1] = psi_a
        r_preds[:, rep] = r̂
    end

    _finalize_pliv!(m, all_coef, all_se, psi_arr, psi_d_arr, store_predictions,
                    Dict("ml_r" => r_preds); var_scaling=var_scaling)
end

function _fit_pliv_partial_xz!(m::DoubleMLPLIV; store_predictions::Bool=true)
    data = m.data
    X, y, d = data.x, data.y, data.d
    Z = data.z
    XZ = hcat(X, Z)
    n = n_obs(data)
    n_rep = m.n_rep
    _pliv_ensure_smpls!(m)

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    l_preds = fill(NaN, n, n_rep)
    m_preds = fill(NaN, n, n_rep)
    r_preds = fill(NaN, n, n_rep)
    var_scaling = fill(Float64(n), 1)

    for rep in 1:n_rep
        folds = m.smpls[rep]
        ℓ̂ = cross_fit_predict(m.ml_l, X, y, folds; classifier=false)
        m̂, train_list = _cross_fit_with_train_preds(m.ml_m, XZ, d, folds)
        m̃ = _cross_fit_fold_targets(m.ml_r, X, train_list, folds)

        u = y .- ℓ̂
        w = d .- m̃
        v = m̂ .- m̃
        psi_a = -(w .* v)
        psi_b = v .* u

        θ = est_coef_linear(psi_a, psi_b)
        se, vsf = _pliv_se(m, psi_a, psi_b, θ, folds, rep)
        all_coef[1, rep] = θ
        all_se[1, rep] = se
        var_scaling[1] = vsf
        psi_arr[:, rep, 1] = psi_a .* θ .+ psi_b
        psi_d_arr[:, rep, 1] = psi_a
        l_preds[:, rep] = ℓ̂
        m_preds[:, rep] = m̂
        r_preds[:, rep] = m̃
    end

    _finalize_pliv!(m, all_coef, all_se, psi_arr, psi_d_arr, store_predictions,
                    Dict("ml_l" => l_preds, "ml_m" => m_preds, "ml_r" => r_preds);
                    var_scaling=var_scaling)
end

function _finalize_pliv!(m, all_coef, all_se, psi_arr, psi_d_arr, store_predictions, preds;
                         var_scaling=nothing)
    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.var_scaling = var_scaling
    m.boot = nothing
    if is_cluster_data(m.data)
        m.cluster_dict = (
            smpls=m.smpls, smpls_cluster=m.smpls_cluster,
            cluster_vars=m.data.cluster, n_folds_per_cluster=m.n_folds_per_cluster,
        )
    end
    if store_predictions
        # drop unused ml_g empty unless IV-type
        if m.score == "IV-type" && haskey(preds, "ml_g")
            m.predictions = preds
        else
            delete!(preds, "ml_g")
            m.predictions = preds
        end
    end
    m.fitted = true
    return m
end
