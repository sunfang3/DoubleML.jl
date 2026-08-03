# Partially Linear Panel Regression (PLPR)
# Python: doubleml.DoubleMLPLPR (Clarke & Polselli 2025)
#
# Approaches:
# - fd_exact: first-difference ΔY, ΔD with covariates (X_t, X_{t-1})
# - wg_approx: within demeaning (y,d,x) + grand mean
# - cre_general: Mundlak/CRE — append unit means of X; adjust m̂ by d̄_i − m̄_i
# - cre_normal: CRE with m fitted on (X, X̄, d̄)

"""
    DoubleMLPLPR

Partially linear panel regression with static-panel approaches.

# Approaches
| `approach` | Transform |
|------------|-----------|
| `"fd_exact"` | First differences of Y,D; covariates `(X_t, X_{t-1})` |
| `"wg_approx"` | Within demeaning of Y,D,X (plus grand mean) |
| `"cre_general"` | Levels with unit means of X; CRE adjustment of `m̂` |
| `"cre_normal"` | Levels with unit means of X; `m` fit on `(X,X̄,d̄)` |

# Data
Long panel in [`DoubleMLData`](@ref) with `id` and `t`.
`d` is the time-varying treatment.
"""
mutable struct DoubleMLPLPR <: AbstractDoubleML
    data::DoubleMLData
    ml_l::Any
    ml_m::Any
    ml_g::Any
    approach::String
    score::String
    n_folds::Int
    n_rep::Int
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
    fitted::Bool
    rng::AbstractRNG
    # transformed cross-section / panel used for estimation
    transformed::Union{Nothing,DoubleMLData}
    d_mean::Union{Nothing,Vector{Float64}}
    transform_id::Union{Nothing,Vector{Int}}
end

const _PLPR_APPROACHES = ("fd_exact", "wg_approx", "cre_general", "cre_normal")

function DoubleMLPLPR(data::DoubleMLData, ml_l, ml_m;
                      ml_g=nothing,
                      approach::AbstractString="fd_exact",
                      score::AbstractString="partialling out",
                      n_folds::Int=5,
                      n_rep::Int=1,
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    data.id === nothing && throw(ArgumentError("PLPR requires id"))
    data.t === nothing && throw(ArgumentError("PLPR requires t"))
    ap = String(approach)
    ap in _PLPR_APPROACHES ||
        throw(ArgumentError("approach must be one of $(_PLPR_APPROACHES)"))
    sc = String(score)
    sc in ("partialling out", "IV-type") ||
        throw(ArgumentError("score must be \"partialling out\" or \"IV-type\""))
    if sc == "IV-type" && ml_g === nothing
        ml_g = clone(ml_l)
    end

    smpls = Vector{Any}()
    return DoubleMLPLPR(
        data, ml_l, ml_m, ml_g, ap, sc, n_folds, n_rep, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        Array{Float64,3}(undef, 0, 0, 0), Array{Float64,3}(undef, 0, 0, 0),
        Dict{String,Any}(), [data.d_col],
        nothing, false, rng, nothing, nothing, nothing,
    )
end

# ---- panel transforms ------------------------------------------------------

function _panel_row_map(data::DoubleMLData)
    row_of = Dict{Tuple{Int,Int},Int}()
    for i in 1:length(data.id)
        row_of[(data.id[i], data.t[i])] = i
    end
    units = sort(unique(data.id))
    times = sort(unique(data.t))
    return row_of, units, times
end

function _unit_means(data::DoubleMLData)
    # mean of x and d by id (over available periods)
    um_x = Dict{Int,Vector{Float64}}()
    um_d = Dict{Int,Float64}()
    cnt = Dict{Int,Int}()
    p = size(data.x, 2)
    for i in 1:n_obs(data)
        u = data.id[i]
        if !haskey(um_x, u)
            um_x[u] = zeros(p)
            um_d[u] = 0.0
            cnt[u] = 0
        end
        um_x[u] .+= vec(data.x[i, :])
        um_d[u] += data.d[i]
        cnt[u] += 1
    end
    for u in keys(um_x)
        um_x[u] ./= cnt[u]
        um_d[u] /= cnt[u]
    end
    return um_x, um_d
end

"""First-difference the panel: one row per consecutive (id, t) pair."""
function _first_difference_panel(data::DoubleMLData)
    row_of, units, times = _panel_row_map(data)
    length(times) >= 2 || error("Need ≥2 periods for first difference")
    dY = Float64[]; dD = Float64[]
    Xrows = Vector{Vector{Float64}}()
    for u in units
        for k in 2:length(times)
            t0, t1 = times[k - 1], times[k]
            haskey(row_of, (u, t0)) || continue
            haskey(row_of, (u, t1)) || continue
            i0 = row_of[(u, t0)]; i1 = row_of[(u, t1)]
            push!(dY, data.y[i1] - data.y[i0])
            push!(dD, data.d[i1] - data.d[i0])
            push!(Xrows, vcat(vec(data.x[i1, :]), vec(data.x[i0, :])))
        end
    end
    isempty(dY) && error("No first-differenced observations")
    X = reduce(vcat, (r' for r in Xrows))
    return DoubleMLData(X, dY, dD; y_col="dy", d_col="dd"), nothing, nothing
end

"""Within demeaning (WG approx): y* = y − ȳ_i + ȳ, etc."""
function _within_demean_panel(data::DoubleMLData)
    um_x, um_d = _unit_means(data)
    n = n_obs(data)
    gy = mean(data.y); gd = mean(data.d)
    gx = vec(mean(data.x; dims=1))
    um_y = Dict{Int,Float64}()
    cnt = Dict{Int,Int}()
    for i in 1:n
        u = data.id[i]
        um_y[u] = get(um_y, u, 0.0) + data.y[i]
        cnt[u] = get(cnt, u, 0) + 1
    end
    for u in keys(um_y)
        um_y[u] /= cnt[u]
    end
    ydm = similar(data.y)
    ddm = similar(data.d)
    Xdm = similar(data.x)
    for i in 1:n
        u = data.id[i]
        ydm[i] = data.y[i] - um_y[u] + gy
        ddm[i] = data.d[i] - um_d[u] + gd
        Xdm[i, :] = vec(data.x[i, :]) .- um_x[u] .+ gx
    end
    return DoubleMLData(Xdm, ydm, ddm; y_col="y_dm", d_col="d_dm"), nothing, nothing
end

"""CRE levels: covariates (X, X̄_i); return d̄_i and id per row."""
function _cre_panel(data::DoubleMLData)
    um_x, um_d = _unit_means(data)
    n = n_obs(data)
    p = size(data.x, 2)
    Xaug = zeros(n, 2p)
    d_mean = zeros(n)
    ids = zeros(Int, n)
    for i in 1:n
        u = data.id[i]
        Xaug[i, 1:p] = data.x[i, :]
        Xaug[i, p+1:2p] = um_x[u]
        d_mean[i] = um_d[u]
        ids[i] = u
    end
    td = DoubleMLData(Xaug, data.y, data.d; y_col=data.y_col, d_col=data.d_col, id=ids, t=data.t)
    return td, d_mean, ids
end

function _transform_plpr(data::DoubleMLData, approach::String)
    if approach == "fd_exact"
        return _first_difference_panel(data)
    elseif approach == "wg_approx"
        return _within_demean_panel(data)
    else
        return _cre_panel(data)
    end
end

# ---- estimation ------------------------------------------------------------

function fit!(m::DoubleMLPLPR; store_predictions::Bool=true)
    td, d_mean, ids = _transform_plpr(m.data, m.approach)
    m.transformed = td
    m.d_mean = d_mean
    m.transform_id = ids

    X, y, d = td.x, td.y, td.d
    n = n_obs(td)
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
        ℓ̂ = cross_fit_predict(m.ml_l, X, y, folds; classifier=false)

        # m nuisance
        if m.approach == "cre_normal"
            Xm = hcat(X, d_mean)
            m̂ = cross_fit_predict(m.ml_m, Xm, d, folds; classifier=is_classifier(m.ml_m))
        else
            m̂ = cross_fit_predict(m.ml_m, X, d, folds; classifier=is_classifier(m.ml_m))
        end
        if m.approach == "cre_general"
            # m* = m̂ + d̄_i − mean_i(m̂)
            m_bar = _group_mean_by_id(m̂, ids)
            m̂ = m̂ .+ d_mean .- m_bar
        end

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
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("ml_l" => l_preds, "ml_m" => m_preds)
        m.score == "IV-type" && (m.predictions["ml_g"] = g_preds)
    end
    m.fitted = true
    return m
end

function _group_mean_by_id(v::AbstractVector, ids::AbstractVector{<:Integer})
    sums = Dict{Int,Float64}()
    cnt = Dict{Int,Int}()
    for i in eachindex(v)
        u = ids[i]
        sums[u] = get(sums, u, 0.0) + v[i]
        cnt[u] = get(cnt, u, 0) + 1
    end
    out = similar(v, Float64)
    for i in eachindex(v)
        u = ids[i]
        out[i] = sums[u] / cnt[u]
    end
    return out
end
