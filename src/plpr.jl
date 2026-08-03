# Partially Linear Panel Regression (PLPR) — first-difference exact approach
# Python: doubleml.DoubleMLPLPR with approach='fd_exact' (Clarke & Polselli 2025)
#
# Transform: ΔY_it = θ ΔD_it + Δg(X) + ΔU
# Estimated as PLR residual-on-residual on first-differenced data with
# covariates (X_t, X_{t-1}).

"""
    DoubleMLPLPR

Partially linear panel regression via **first differences** (`approach=:fd_exact`).

# Data
Long panel in [`DoubleMLData`](@ref) with `id` and `t`.
`d` is the time-varying treatment (not group label).
"""
mutable struct DoubleMLPLPR <: AbstractDoubleML
    data::DoubleMLData
    ml_l::Any
    ml_m::Any
    approach::String
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
    # internal FD cross-section
    fd_data::Union{Nothing,DoubleMLData}
end

function DoubleMLPLPR(data::DoubleMLData, ml_l, ml_m;
                      approach::AbstractString="fd_exact",
                      n_folds::Int=5,
                      n_rep::Int=1,
                      draw_sample_splitting::Bool=true,
                      rng::AbstractRNG=Random.default_rng())
    data.id === nothing && throw(ArgumentError("PLPR requires id"))
    data.t === nothing && throw(ArgumentError("PLPR requires t"))
    approach == "fd_exact" ||
        throw(ArgumentError("Only approach=\"fd_exact\" implemented (CRE/WG later)"))

    # sample splitting is rebuilt on the first-differenced cross-section in fit!
    smpls = Vector{Any}()
    return DoubleMLPLPR(
        data, ml_l, ml_m, String(approach), n_folds, n_rep, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        Array{Float64,3}(undef, 0, 0, 0), Array{Float64,3}(undef, 0, 0, 0),
        Dict{String,Any}(), [data.d_col],
        nothing, false, rng, nothing,
    )
end

"""First-difference the panel: one row per (id, t) with t≥2."""
function _first_difference_panel(data::DoubleMLData)
    ids = data.id
    ts = data.t
    row_of = Dict{Tuple{Int,Int},Int}()
    for i in 1:length(ids)
        row_of[(ids[i], ts[i])] = i
    end
    units = unique(ids)
    times = sort(unique(ts))
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
            # covariates: [X_t, X_{t-1}]
            push!(Xrows, vcat(vec(data.x[i1, :]), vec(data.x[i0, :])))
        end
    end
    isempty(dY) && error("No first-differenced observations")
    X = reduce(vcat, (r' for r in Xrows))
    return DoubleMLData(X, dY, dD; y_col="dy", d_col="dd")
end

function fit!(m::DoubleMLPLPR; store_predictions::Bool=true)
    fd = _first_difference_panel(m.data)
    m.fd_data = fd
    # PLR partialling-out on FD data
    plr = DoubleMLPLR(fd, clone(m.ml_l), clone(m.ml_m);
                      n_folds=m.n_folds, n_rep=m.n_rep, rng=m.rng)
    fit!(plr; store_predictions=store_predictions)
    m.coef = plr.coef
    m.se = plr.se
    m.all_coef = plr.all_coef
    m.all_se = plr.all_se
    m.psi = plr.psi
    m.psi_deriv = plr.psi_deriv
    m.predictions = plr.predictions
    m.boot = nothing
    m.fitted = true
    m.smpls = plr.smpls
    return m
end
