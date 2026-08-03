# Multi-period staggered DiD (simplified Callaway–Sant'Anna style)
# Builds group–time ATTs via two-period DiD scores on panel long data.

"""
    DoubleMLPanelData

Long panel: one row per (id, t) with `y`, treatment `d` (or first-treatment group
stored in `d` as first treated period / 0=never), covariates `x`.
"""
const DoubleMLPanelData = DoubleMLData  # constructed with id= and t=

"""
    DoubleMLDIDMulti

Group–time ATTs for staggered adoption panels.

For each first-treatment group `g` and evaluation time `t ≥ g`, estimates
`ATT(g,t)` by comparing group `g` to never-treated units on the outcome change
from `t_pre = g-1` to `t`, using the observational DiD score.

# Data
Provide long panel in [`DoubleMLData`](@ref) with:
- `id`, `t` set
- `d` = first treatment period (`0` or `Inf`-like large number for never treated;
  use `0` for never treated)
- `y` = outcome level at time `t`
"""
mutable struct DoubleMLDIDMulti <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    control_group::String
    anticipation_periods::Int
    n_folds::Int
    n_rep::Int
    trimming_threshold::Float64
    gt_combos::Vector{NTuple{3,Int}}  # (g, t_pre, t_eval)
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
    att_models::Vector{DoubleMLDID}
    fitted::Bool
    rng::AbstractRNG
end

function _never_treated_code(dvals)
    # never treated stored as 0
    0
end

function _standard_gt_combinations(groups, times; anticipation::Int=0)
    never = 0
    gs = sort(unique(filter(g -> g != never, groups)))
    ts = sort(unique(times))
    combos = NTuple{3,Int}[]
    for g in gs
        for t_eval in ts
            t_eval < g && continue  # only post (and on-impact)
            t_pre = g - 1 - anticipation
            t_pre in ts || continue
            push!(combos, (Int(g), Int(t_pre), Int(t_eval)))
        end
    end
    return combos
end

function DoubleMLDIDMulti(data::DoubleMLData, ml_g, ml_m;
                          control_group::AbstractString="never_treated",
                          anticipation_periods::Int=0,
                          gt_combinations=:standard,
                          n_folds::Int=5,
                          n_rep::Int=1,
                          trimming_threshold::Real=1e-2,
                          rng::AbstractRNG=Random.default_rng())
    data.id === nothing && throw(ArgumentError("panel data requires id"))
    data.t === nothing && throw(ArgumentError("panel data requires t"))
    control_group == "never_treated" ||
        throw(ArgumentError("only control_group=\"never_treated\" supported"))

    combos = if gt_combinations === :standard
        _standard_gt_combinations(data.d, data.t; anticipation=anticipation_periods)
    else
        NTuple{3,Int}[Tuple(c) for c in gt_combinations]
    end
    isempty(combos) && throw(ArgumentError("no (g,t_pre,t_eval) combinations found"))

    n_att = length(combos)
    names = ["ATT(g=$g,t_pre=$tp,t=$te)" for (g, tp, te) in combos]
    # psi length will be n_units (unique ids) — store after fit
    return DoubleMLDIDMulti(
        data, ml_g, ml_m, String(control_group), anticipation_periods,
        n_folds, n_rep, Float64(trimming_threshold), combos,
        Vector{Any}(),
        Float64[], Float64[],
        zeros(n_att, n_rep), zeros(n_att, n_rep),
        Array{Float64,3}(undef, 0, 0, 0), Array{Float64,3}(undef, 0, 0, 0),
        Dict{String,Any}(), names,
        nothing, DoubleMLDID[], false, rng,
    )
end

"""Build cross-section of ΔY and group indicator for one (g,t_pre,t_eval)."""
function _did_multi_cs(data::DoubleMLData, g::Int, t_pre::Int, t_eval::Int)
    ids = data.id
    ts = data.t
    # map (id,t) -> row
    row_of = Dict{Tuple{Int,Int},Int}()
    for i in 1:length(ids)
        row_of[(ids[i], ts[i])] = i
    end
    never = 0
    # unique units in group g or never, with both times observed
    units = unique(ids)
    Xrows = Vector{Vector{Float64}}()
    ydelta = Float64[]
    dgroup = Float64[]
    for u in units
        # group label from any row
        rows_u = findall(==(u), ids)
        g_u = Int(round(data.d[rows_u[1]]))
        (g_u == g || g_u == never) || continue
        haskey(row_of, (u, t_pre)) || continue
        haskey(row_of, (u, t_eval)) || continue
        i0 = row_of[(u, t_pre)]
        i1 = row_of[(u, t_eval)]
        push!(ydelta, data.y[i1] - data.y[i0])
        push!(dgroup, g_u == g ? 1.0 : 0.0)
        # covariates from pre period
        push!(Xrows, vec(data.x[i0, :]))
    end
    isempty(ydelta) && error("No units for ATT(g=$g, t_pre=$t_pre, t=$t_eval)")
    X = reduce(vcat, [r' for r in Xrows])
    return DoubleMLData(X, ydelta, dgroup)
end

function fit!(m::DoubleMLDIDMulti; store_predictions::Bool=false)
    n_att = length(m.gt_combos)
    n_rep = m.n_rep
    all_coef = zeros(n_att, n_rep)
    all_se = zeros(n_att, n_rep)
    m.att_models = DoubleMLDID[]

    max_n = 0
    psi_list = Matrix{Float64}[]  # each n_u × n_rep for one ATT — pad later
    psi_d_list = Matrix{Float64}[]

    for (j, (g, t_pre, t_eval)) in enumerate(m.gt_combos)
        cs = _did_multi_cs(m.data, g, t_pre, t_eval)
        did = DoubleMLDID(
            cs, clone(m.ml_g), clone(m.ml_m);
            n_folds=m.n_folds, n_rep=n_rep,
            score="observational",
            trimming_threshold=m.trimming_threshold,
            rng=copy(m.rng),
        )
        fit!(did; store_predictions=store_predictions)
        push!(m.att_models, did)
        all_coef[j, :] = did.all_coef[1, :]
        all_se[j, :] = did.all_se[1, :]
        max_n = max(max_n, size(did.psi, 1))
        push!(psi_list, did.psi[:, :, 1])
        push!(psi_d_list, did.psi_deriv[:, :, 1])
    end

    # pad psi to max_n × n_rep × n_att
    psi = fill(0.0, max_n, n_rep, n_att)
    psi_d = fill(-1.0, max_n, n_rep, n_att)
    for j in 1:n_att
        n_j = size(psi_list[j], 1)
        psi[1:n_j, :, j] = psi_list[j]
        psi_d[1:n_j, :, j] = psi_d_list[j]
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi; m.psi_deriv = psi_d
    m.boot = nothing
    m.fitted = true
    return m
end
