# Multi-period staggered DiD — Callaway & Sant'Anna (2021) toolbox
# Aligned with Python doubleml.did.DoubleMLDIDMulti
#
# Features:
# - Group–time ATTs ATT(g, t_pre, t_eval)
# - Control groups: never_treated | not_yet_treated
# - Anticipation periods
# - gt_combinations: standard | all | universal | explicit list
# - Aggregation: group | time | eventstudy | overall
# - Results table with event time e = t_eval − g

"""
    DoubleMLPanelData

Alias: long panel via [`DoubleMLData`](@ref) with `id` and `t` set.
`d` stores first-treatment period (`0` = never treated).
"""
const DoubleMLPanelData = DoubleMLData

# ---- helpers ----------------------------------------------------------------

"""Never-treated code: largest non-positive sentinel; we use `0`."""
_never_treated_value() = 0

_is_never_treated(g, never) = isapprox(g, never; atol=0) || g < 0

function _g_values(data::DoubleMLData)
    sort(unique(Int.(round.(data.d))))
end

function _t_values(data::DoubleMLData)
    sort(unique(Int.(data.t)))
end

function _unit_g_map(data::DoubleMLData)
    # id → first-treatment group (from any row)
    m = Dict{Int,Int}()
    for i in 1:length(data.id)
        u = data.id[i]
        if !haskey(m, u)
            m[u] = Int(round(data.d[i]))
        end
    end
    return m
end

"""
Construct (g, t_pre, t_eval) combinations.

- `:standard` — CS default: base period last pre-treatment (adj. anticipation)
- `:all` — all valid (g, t_pre, t_eval) with t_pre before treatment
- `:universal` — fixed base period for each g; all other t as eval
"""
function _construct_gt_combinations(setting::Symbol, g_values, t_values, never, anticipation::Int)
    treatment_groups = sort([g for g in g_values if !_is_never_treated(g, never)])
    isempty(treatment_groups) && throw(ArgumentError("no treated groups found (all never-treated?)"))
    t_values = sort(collect(t_values))
    combos = NTuple{3,Int}[]

    for g_val in treatment_groups
        t_before = t_values[t_values .< g_val]
        length(t_before) <= anticipation && continue
        first_eval_index = anticipation + 1  # 1-based offset into t_values for first eval slot

        if setting === :standard
            t_before_g = t_before[end - anticipation]  # last relevant pre base
            # enumerate t_eval from t_values[first_eval_index+1] ... actually Python uses
            # t_values[first_eval_index:] with 0-based first_eval_index = anticipation+1
            # so Julia: t_values[(anticipation+2):end] if 1-based... 
            # Python: first_eval_index = anticipation_periods + 1  (0-based index)
            # t_values[first_eval_index:] → from element at 0-based index anticipation+1
            # = Julia index anticipation+2
            start_idx = anticipation + 2
            start_idx > length(t_values) && continue
            for (i_off, t_eval) in enumerate(t_values[start_idx:end])
                # i_t_eval in Python is 0 for first element of t_values[first_eval_index:]
                # t_values[i_t_eval] in that slice... 
                # Python: min(t_values[i_t_eval], t_before_g) where i_t_eval indexes full t_values
                # for i_t_eval, t_eval in enumerate(t_values[first_eval_index:]):
                #   i_t_eval is 0,1,2,... but min(t_values[i_t_eval], ...) uses 0-based full index WRONG?
                # Looking again carefully:
                #   for i_t_eval, t_eval in enumerate(t_values[first_eval_index:]):
                #       (g_val, min(t_values[i_t_eval], t_before_g), t_eval)
                # Here i_t_eval is 0,1,2... so t_values[0], t_values[1] — that's a Python bug or
                # enumerate of the slice means i_t_eval is relative...
                # In Python, enumerate(arr) gives indices 0..len-1 of the slice, and
                # t_values[i_t_eval] uses those as indices into FULL t_values.
                # So for first_eval_index=1, t_values[1:]=[t1,t2,t3], enumerate gives
                # i=0,t=t1 → min(t_values[0], t_before)=min(t0, t_before)
                # i=1,t=t2 → min(t_values[1], t_before)
                # This seems intentional for pre-period t_pre selection.
                i_abs = start_idx + i_off - 1  # 1-based index of t_eval in t_values
                # match Python: i_t_eval = i_off - 1 (0-based relative) used as full index
                i_py = i_off - 1  # 0-based
                t_pre = min(t_values[i_py + 1], t_before_g)  # Julia 1-based
                push!(combos, (Int(g_val), Int(t_pre), Int(t_eval)))
            end

        elseif setting === :all
            start_idx = anticipation + 2
            start_idx > length(t_values) && continue
            for t_eval in t_values[start_idx:end]
                # t_pre candidates: t_values[t_values <= min(g,t_eval)] excluding last `anticipation+1`
                tmax = min(g_val, t_eval)
                candidates = t_values[t_values .<= tmax]
                length(candidates) <= anticipation + 1 && continue
                for t_pre in candidates[1:(end - anticipation - 1)]
                    push!(combos, (Int(g_val), Int(t_pre), Int(t_eval)))
                end
            end

        elseif setting === :universal
            base = t_before[end - anticipation]
            for t_eval in t_values
                t_eval == base && continue
                push!(combos, (Int(g_val), Int(base), Int(t_eval)))
            end
        else
            throw(ArgumentError("gt_combinations must be :standard, :all, or :universal"))
        end
    end
    isempty(combos) && throw(ArgumentError("No valid group-time combinations found"))
    return combos
end

# ---- DIDMulti model ---------------------------------------------------------

"""
    DoubleMLDIDMulti

Callaway–Sant'Anna multi-period staggered DiD with double machine learning.

# Arguments
- `control_group`: `"never_treated"` (default) or `"not_yet_treated"`
- `anticipation_periods`: periods of anticipation (default `0`)
- `gt_combinations`: `:standard` | `:all` | `:universal` | vector of `(g,t_pre,t_eval)`

# After `fit!`
- `summary_table(m)` — group–time ATTs
- `aggregate(m, :group|:time|:eventstudy)` — CS-style aggregations
- `att_table(m)` — DataFrame with `g, t_pre, t_eval, event_time, coef, se, ...`
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
    in_sample_normalization::Bool
    never_treated_value::Int
    g_values::Vector{Int}
    t_values::Vector{Int}
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
    unit_ids_per_att::Vector{Vector{Int}}  # unit ids used in each ATT
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLDIDMulti(data::DoubleMLData, ml_g, ml_m;
                          control_group::AbstractString="never_treated",
                          anticipation_periods::Int=0,
                          gt_combinations=:standard,
                          n_folds::Int=5,
                          n_rep::Int=1,
                          trimming_threshold::Real=1e-2,
                          in_sample_normalization::Bool=true,
                          never_treated_value::Integer=0,
                          rng::AbstractRNG=Random.default_rng())
    data.id === nothing && throw(ArgumentError("panel data requires id"))
    data.t === nothing && throw(ArgumentError("panel data requires t"))
    cg = String(control_group)
    cg in ("never_treated", "not_yet_treated") ||
        throw(ArgumentError("control_group must be \"never_treated\" or \"not_yet_treated\""))
    anticipation_periods >= 0 || throw(ArgumentError("anticipation_periods ≥ 0"))

    never = Int(never_treated_value)
    gvals = _g_values(data)
    tvals = _t_values(data)

    combos = if gt_combinations isa Symbol
        _construct_gt_combinations(gt_combinations, gvals, tvals, never, anticipation_periods)
    else
        NTuple{3,Int}[Tuple(Int.(c)) for c in gt_combinations]
    end

    n_att = length(combos)
    names = ["ATT(g=$g,t_pre=$tp,t=$te)" for (g, tp, te) in combos]
    return DoubleMLDIDMulti(
        data, ml_g, ml_m, cg, anticipation_periods,
        n_folds, n_rep, Float64(trimming_threshold), in_sample_normalization,
        never, gvals, tvals, combos,
        Vector{Any}(),
        Float64[], Float64[],
        zeros(n_att, n_rep), zeros(n_att, n_rep),
        Array{Float64,3}(undef, 0, 0, 0), Array{Float64,3}(undef, 0, 0, 0),
        Dict{String,Any}(), names,
        nothing, DoubleMLDID[], Vector{Int}[],
        false, rng,
    )
end

"""
Build cross-section (ΔY, G, X) for one (g, t_pre, t_eval) with control group selection.
Returns `(data::DoubleMLData, unit_ids::Vector{Int})`.
"""
function _did_multi_cs(data::DoubleMLData, g::Int, t_pre::Int, t_eval::Int,
                       control_group::String, never::Int, anticipation::Int, t_values::Vector{Int})
    ids = data.id
    ts = data.t
    row_of = Dict{Tuple{Int,Int},Int}()
    for i in 1:length(ids)
        row_of[(ids[i], ts[i])] = i
    end
    ug = _unit_g_map(data)

    # max_g_value for not_yet_treated (Python logic)
    comparison_period = max(t_eval, g)
    idx_c = findfirst(==(comparison_period), t_values)
    max_g = if idx_c === nothing
        comparison_period
    else
        t_values[min(idx_c + anticipation, length(t_values))]
    end

    Xrows = Vector{Vector{Float64}}()
    ydelta = Float64[]
    dgroup = Float64[]
    unit_ids = Int[]

    for u in sort(collect(keys(ug)))
        g_u = ug[u]
        G_ind = g_u == g
        if control_group == "never_treated"
            C_ind = _is_never_treated(g_u, never)
        else
            later = (g_u > max_g) && !G_ind
            C_ind = _is_never_treated(g_u, never) || later
        end
        (G_ind || C_ind) || continue
        G_ind && C_ind && continue  # disjoint
        haskey(row_of, (u, t_pre)) || continue
        haskey(row_of, (u, t_eval)) || continue
        i0 = row_of[(u, t_pre)]
        i1 = row_of[(u, t_eval)]
        push!(ydelta, data.y[i1] - data.y[i0])
        push!(dgroup, G_ind ? 1.0 : 0.0)
        push!(Xrows, vec(data.x[i0, :]))
        push!(unit_ids, u)
    end
    isempty(ydelta) && error("No observations for ATT(g=$g, t_pre=$t_pre, t=$t_eval)")
    sum(dgroup) < 1 && error("No treated units for ATT(g=$g, t=$t_eval)")
    sum(dgroup .== 0) < 1 && error("No control units for ATT(g=$g, t=$t_eval; control=$control_group)")
    X = reduce(vcat, (r' for r in Xrows))
    return DoubleMLData(X, ydelta, dgroup), unit_ids
end

function fit!(m::DoubleMLDIDMulti; store_predictions::Bool=false)
    n_att = length(m.gt_combos)
    n_rep = m.n_rep
    all_coef = zeros(n_att, n_rep)
    all_se = zeros(n_att, n_rep)
    m.att_models = DoubleMLDID[]
    m.unit_ids_per_att = Vector{Int}[]

    psi_list = Matrix{Float64}[]
    psi_d_list = Matrix{Float64}[]
    max_n = 0

    for (j, (g, t_pre, t_eval)) in enumerate(m.gt_combos)
        cs, uids = _did_multi_cs(
            m.data, g, t_pre, t_eval,
            m.control_group, m.never_treated_value, m.anticipation_periods, m.t_values,
        )
        did = DoubleMLDID(
            cs, clone(m.ml_g), clone(m.ml_m);
            n_folds=m.n_folds, n_rep=n_rep,
            score="observational",
            in_sample_normalization=m.in_sample_normalization,
            trimming_threshold=m.trimming_threshold,
            rng=copy(m.rng),
        )
        fit!(did; store_predictions=store_predictions)
        push!(m.att_models, did)
        push!(m.unit_ids_per_att, uids)
        all_coef[j, :] = did.all_coef[1, :]
        all_se[j, :] = did.all_se[1, :]
        max_n = max(max_n, size(did.psi, 1))
        push!(psi_list, did.psi[:, :, 1])
        push!(psi_d_list, did.psi_deriv[:, :, 1])
    end

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

"""Group–time ATT table with event time `e = t_eval − g`."""
function att_table(m::DoubleMLDIDMulti; level::Real=0.95)
    m.fitted || error("Call fit! first")
    z = quantile(Normal(), 1 - (1 - level) / 2)
    tstat = m.coef ./ m.se
    p = 2 .* cdf.(Normal(), -abs.(tstat))
    g = [c[1] for c in m.gt_combos]
    t_pre = [c[2] for c in m.gt_combos]
    t_eval = [c[3] for c in m.gt_combos]
    e = t_eval .- g
    return DataFrame(
        g = g,
        t_pre = t_pre,
        t_eval = t_eval,
        event_time = e,
        coef = m.coef,
        std_err = m.se,
        t = tstat,
        pvalue = p,
        ci_lower = m.coef .- z .* m.se,
        ci_upper = m.coef .+ z .* m.se,
        post = e .>= 0,
    )
end

function summary_table(m::DoubleMLDIDMulti; level::Real=0.95)
    att_table(m; level=level)
end

# ---- Aggregation (Callaway–Sant'Anna) ---------------------------------------

"""Result of [`aggregate`](@ref) for DID multi."""
struct DIDAggregation
    method::String
    names::Vector{String}
    coef::Vector{Float64}
    se::Vector{Float64}
    weights_overall::Vector{Float64}  # weight of each aggregation in overall
    overall_coef::Float64
    overall_se::Float64
end

function Base.show(io::IO, a::DIDAggregation)
    print(io, "DIDAggregation($(a.method); overall=$(round(a.overall_coef; digits=4)), ",
          "n=$(length(a.coef)))")
end

function summary_table(a::DIDAggregation; level::Real=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    tstat = a.coef ./ a.se
    p = 2 .* cdf.(Normal(), -abs.(tstat))
    df = DataFrame(
        name = a.names,
        coef = a.coef,
        std_err = a.se,
        t = tstat,
        pvalue = p,
        ci_lower = a.coef .- z .* a.se,
        ci_upper = a.coef .+ z .* a.se,
    )
    # append overall row
    to = a.overall_coef / a.overall_se
    po = 2 * cdf(Normal(), -abs(to))
    push!(df, (
        name = "overall",
        coef = a.overall_coef,
        std_err = a.overall_se,
        t = to,
        pvalue = po,
        ci_lower = a.overall_coef - z * a.overall_se,
        ci_upper = a.overall_coef + z * a.overall_se,
    ))
    return df
end

"""Share of units in each treatment group (balanced panel unit-level)."""
function _group_shares(data::DoubleMLData, g_values::Vector{Int}, never::Int)
    ug = _unit_g_map(data)
    n = length(ug)
    shares = Dict{Int,Float64}()
    for g in g_values
        _is_never_treated(g, never) && continue
        shares[g] = count(==(g), values(ug)) / n
    end
    return shares
end

"""
    aggregate(m::DoubleMLDIDMulti, method=:group; post_only=true) -> DIDAggregation

Callaway–Sant'Anna aggregations of group–time ATTs:

| `method` | Meaning |
|----------|---------|
| `:group` | Average ATT within each first-treatment group `g` (post periods) |
| `:time` | Average ATT at each calendar time `t_eval` |
| `:eventstudy` | Average ATT by event time `e = t_eval − g` |

`post_only=true` (default) restricts to `e ≥ 0` for group/time (eventstudy always
includes pre periods when estimated). Overall ATT is the share-weighted mean of
group aggregations (or post event-time aggregations).
"""
function aggregate(m::DoubleMLDIDMulti, method::Symbol=:group; post_only::Bool=true)
    m.fitted || error("Call fit! before aggregate")
    method in (:group, :time, :eventstudy) ||
        throw(ArgumentError("method must be :group, :time, or :eventstudy"))

    n_att = length(m.gt_combos)
    gvec = [c[1] for c in m.gt_combos]
    tpre = [c[2] for c in m.gt_combos]
    teval = [c[3] for c in m.gt_combos]
    evec = teval .- gvec
    shares = _group_shares(m.data, m.g_values, m.never_treated_value)

    # select ATT indices
    if method === :eventstudy
        sel = trues(n_att)
    else
        sel = post_only ? (evec .>= 0) : trues(n_att)
    end
    any(sel) || error("No ATT cells selected for aggregation")

    if method === :group
        groups = sort(unique(gvec[sel]))
        names = String[]; coefs = Float64[]; ses = Float64[]; w_ov = Float64[]
        for g in groups
            idx = findall(i -> sel[i] && gvec[i] == g, 1:n_att)
            w = fill(1 / length(idx), length(idx))
            c, s = _weighted_att(m, idx, w)
            push!(names, "g=$g")
            push!(coefs, c); push!(ses, s)
            push!(w_ov, get(shares, g, 0.0))
        end
        # normalize overall weights
        sw = sum(w_ov)
        sw > 0 || (w_ov .= 1 / length(w_ov); sw = 1.0)
        w_ov ./= sw
        oc, os = _combine_aggs(coefs, ses, w_ov)
        return DIDAggregation("group", names, coefs, ses, w_ov, oc, os)

    elseif method === :time
        times = sort(unique(teval[sel]))
        names = String[]; coefs = Float64[]; ses = Float64[]
        for t in times
            idx = findall(i -> sel[i] && teval[i] == t, 1:n_att)
            # weight by group share within this t
            ws = Float64[]
            for i in idx
                push!(ws, get(shares, gvec[i], 0.0))
            end
            ssum = sum(ws)
            ssum > 0 || (ws .= 1 / length(ws); ssum = 1.0)
            ws ./= ssum
            c, s = _weighted_att(m, idx, ws)
            push!(names, "t=$t"); push!(coefs, c); push!(ses, s)
        end
        w_ov = fill(1 / length(coefs), length(coefs))
        oc, os = _combine_aggs(coefs, ses, w_ov)
        return DIDAggregation("time", names, coefs, ses, w_ov, oc, os)

    else  # eventstudy
        es = sort(unique(evec[sel]))
        names = String[]; coefs = Float64[]; ses = Float64[]; w_ov = Float64[]
        n_post = count(>=(0), es)
        for e in es
            idx = findall(i -> sel[i] && evec[i] == e, 1:n_att)
            ws = Float64[get(shares, gvec[i], 0.0) for i in idx]
            ssum = sum(ws)
            ssum > 0 || (ws .= 1 / length(ws); ssum = 1.0)
            ws ./= ssum
            c, s = _weighted_att(m, idx, ws)
            push!(names, "e=$e"); push!(coefs, c); push!(ses, s)
            push!(w_ov, e >= 0 ? 1 / max(n_post, 1) : 0.0)
        end
        if sum(w_ov) <= 0
            w_ov .= 1 / length(w_ov)
        else
            w_ov ./= sum(w_ov)
        end
        oc, os = _combine_aggs(coefs, ses, w_ov)
        return DIDAggregation("eventstudy", names, coefs, ses, w_ov, oc, os)
    end
end

function _weighted_att(m::DoubleMLDIDMulti, idx::Vector{Int}, w::Vector{Float64})
    # coef = Σ w_j θ_j
    # se² ≈ Σ w_j² se_j²  (conservative; ignores cross-ATT dependence)
    c = sum(w[k] * m.coef[idx[k]] for k in eachindex(idx))
    v = sum((w[k] * m.se[idx[k]])^2 for k in eachindex(idx))
    return c, sqrt(max(v, 0.0))
end

function _combine_aggs(coefs, ses, w)
    c = sum(w .* coefs)
    v = sum((w .* ses) .^ 2)
    return c, sqrt(max(v, 0.0))
end
