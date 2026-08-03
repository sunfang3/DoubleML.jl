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

"""
    recode_never_treated(d; from=0.0, to=Inf) -> Vector

Map never-treated codes for Python float-panel parity (`0` ↔ `+Inf`).
Works on a treatment-timing vector (first-treatment period).
"""
function recode_never_treated(d::AbstractVector; from::Real=0.0, to::Real=Inf)
    out = Float64.(d)
    if isfinite(from)
        out[isapprox.(out, Float64(from); atol=0)] .= Float64(to)
    else
        out[.!isfinite.(out)] .= Float64(to)
    end
    return out
end

"""
    recode_never_treated(data::DoubleMLData; from=0.0, to=Inf) -> DoubleMLData

Return a copy of panel data with never-treated codes remapped in `d` / `d_mat`.
"""
function recode_never_treated(data::DoubleMLData; from::Real=0.0, to::Real=Inf)
    d_new = recode_never_treated(data.d; from=from, to=to)
    return DoubleMLData(
        data.x, data.y, d_new;
        y_col=data.y_col, d_cols=data.d_cols, x_cols=data.x_cols,
        z=data.z, z_cols=data.z_cols,
        s=data.s, s_col=data.s_col,
        id=data.id, t=data.t, score=data.score,
        cluster=data.cluster, cluster_cols=data.cluster_cols,
        use_other_treat_as_covariate=data.use_other_treat_as_covariate,
    )
end

# ---- helpers ----------------------------------------------------------------

"""
Never-treated coding (Python parity):
- Julia default / integer panels: `0`
- Python float panels often use `+Inf` (or missing → Inf)

`_is_never_treated` accepts `0`, negative, non-finite (`Inf`/`NaN`), or an explicit
`never` sentinel (including `Inf`).
"""
_never_treated_value() = 0.0

function _is_never_treated(g, never)
    # non-finite group codes always count as never-treated (Python +inf)
    !isfinite(Float64(g)) && return true
    if !isfinite(Float64(never))
        return !isfinite(Float64(g))
    end
    return isapprox(Float64(g), Float64(never); atol=0) || Float64(g) < 0
end

"""Detect default never-treated code from data (Inf if any non-finite d)."""
function _detect_never_treated(data::DoubleMLData)
    any(!isfinite, data.d) && return Inf
    return 0.0
end

function _g_values(data::DoubleMLData)
    # finite treatment times only; never-treated may be 0 or Inf
    gs = Float64[]
    for v in unique(data.d)
        if isfinite(v)
            push!(gs, round(v))
        end
    end
    # include never sentinel 0 if present as finite zero
    return sort(unique(Int.(gs)))
end

function _t_values(data::DoubleMLData)
    sort(unique(Int.(data.t)))
end

function _unit_g_map(data::DoubleMLData)
    # id → first-treatment group (from any row); store Float64 to allow Inf
    m = Dict{Int,Float64}()
    for i in 1:length(data.id)
        u = Int(data.id[i])
        if !haskey(m, u)
            m[u] = Float64(data.d[i])
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
    # Mirror Python doubleml.did.did_multi._construct_gt_combinations exactly.
    treatment_groups = sort([g for g in g_values if !_is_never_treated(g, never)])
    isempty(treatment_groups) && throw(ArgumentError("no treated groups found (all never-treated?)"))
    t_values = sort(collect(Int.(t_values)))
    combos = NTuple{3,Int}[]

    for g_val in treatment_groups
        t_before_g_all = t_values[t_values .< g_val]
        # Python: first_eval_index = anticipation_periods + 1  (0-based)
        # skip if not enough pre periods
        length(t_before_g_all) <= anticipation && continue
        first_eval_index_py = anticipation + 1  # 0-based index into t_values
        # Julia 1-based start of t_values[first_eval_index:]
        start_jl = first_eval_index_py + 1
        start_jl > length(t_values) && continue

        if setting === :standard
            # t_before_g = t_values_before_g[-first_eval_index]
            t_before_g = t_before_g_all[end - anticipation]
            # for i_t_eval, t_eval in enumerate(t_values[first_eval_index:]):
            #     (g_val, min(t_values[i_t_eval], t_before_g), t_eval)
            # where i_t_eval is 0-based index into the *slice*, used as index into *full* t_values
            for (i_off, t_eval) in enumerate(t_values[start_jl:end])
                i_py = i_off - 1  # 0-based enumerate index
                t_pre = min(t_values[i_py + 1], t_before_g)
                push!(combos, (Int(g_val), Int(t_pre), Int(t_eval)))
            end

        elseif setting === :all
            # Python:
            # for t_eval in t_values[first_eval_index:]:
            #   for t_pre in t_values[t_values <= min(g_val, t_eval)][:-first_eval_index]:
            for t_eval in t_values[start_jl:end]
                tmax = min(g_val, t_eval)
                candidates = t_values[t_values .<= tmax]
                # drop last `first_eval_index` entries (Python [:-first_eval_index])
                n_drop = first_eval_index_py
                length(candidates) <= n_drop && continue
                for t_pre in candidates[1:(end - n_drop)]
                    push!(combos, (Int(g_val), Int(t_pre), Int(t_eval)))
                end
            end

        elseif setting === :universal
            # base_period = t_values_before_g[-first_eval_index]
            base = t_before_g_all[end - anticipation]
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
    score::String
    trimming_threshold::Float64
    ps_processor::PSProcessor
    in_sample_normalization::Bool
    never_treated_value::Float64
    g_values::Vector{Int}
    t_values::Vector{Int}
    gt_combos::Vector{NTuple{3,Int}}  # (g, t_pre, t_eval)
    smpls::Vector                      # optional unit-level folds (indices into all_unit_ids)
    cell_smpls::Union{Nothing,Vector}  # optional per-ATT smpls (length n_att)
    use_unit_sample_splitting::Bool    # shared unit-level folds for all cells
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
    all_unit_ids::Vector{Int}              # sorted unique panel units
    # unit-aligned influence functions: n_units × n_rep × n_att
    if_units::Array{Float64,3}
    fitted::Bool
    rng::AbstractRNG
end

function DoubleMLDIDMulti(data::DoubleMLData, ml_g, ml_m=nothing;
                          control_group::AbstractString="never_treated",
                          anticipation_periods::Int=0,
                          gt_combinations=:standard,
                          n_folds::Int=5,
                          n_rep::Int=1,
                          score::AbstractString="observational",
                          trimming_threshold::Real=1e-2,
                          ps_processor::Union{Nothing,PSProcessor}=nothing,
                          in_sample_normalization::Bool=true,
                          never_treated_value=nothing,
                          use_unit_sample_splitting::Bool=true,
                          rng::AbstractRNG=Random.default_rng())
    data.id === nothing && throw(ArgumentError("panel data requires id"))
    data.t === nothing && throw(ArgumentError("panel data requires t"))
    cg = String(control_group)
    cg in ("never_treated", "not_yet_treated") ||
        throw(ArgumentError("control_group must be \"never_treated\" or \"not_yet_treated\""))
    anticipation_periods >= 0 || throw(ArgumentError("anticipation_periods ≥ 0"))
    sc = String(score)
    sc in ("observational", "experimental") ||
        throw(ArgumentError("score must be \"observational\" or \"experimental\""))
    sc == "observational" && ml_m === nothing &&
        throw(ArgumentError("ml_m required for score=\"observational\""))

    never = never_treated_value === nothing ?
        _detect_never_treated(data) : Float64(never_treated_value)
    gvals = _g_values(data)
    tvals = _t_values(data)

    combos = if gt_combinations isa Symbol
        _construct_gt_combinations(gt_combinations, gvals, tvals, never, anticipation_periods)
    else
        NTuple{3,Int}[Tuple(Int.(c)) for c in gt_combinations]
    end

    n_att = length(combos)
    names = ["ATT(g=$g,t_pre=$tp,t=$te)" for (g, tp, te) in combos]
    all_uids = sort(unique(Int.(data.id)))
    return DoubleMLDIDMulti(
        data, ml_g, ml_m, cg, anticipation_periods,
        n_folds, n_rep, sc, Float64(trimming_threshold),
        resolve_ps_processor(ps_processor, trimming_threshold),
        in_sample_normalization,
        never, gvals, tvals, combos,
        Vector{Any}(), nothing, use_unit_sample_splitting,
        Float64[], Float64[],
        zeros(n_att, n_rep), zeros(n_att, n_rep),
        Array{Float64,3}(undef, 0, 0, 0), Array{Float64,3}(undef, 0, 0, 0),
        Dict{String,Any}(), names,
        nothing, DoubleMLDID[], Vector{Int}[], all_uids,
        Array{Float64,3}(undef, 0, 0, 0),
        false, rng,
    )
end

"""
    set_sample_splitting!(m::DoubleMLDIDMulti, smpls)

Set **unit-level** sample splits shared across all group–time ATTs.
`smpls` is a vector of length `n_rep`, each entry a vector of
`(train=..., test=...)` folds with indices into `m.all_unit_ids` (1 … n_units).

Enables Python-style shared-panel cross-fitting for numerical alignment.
"""
function set_sample_splitting!(m::DoubleMLDIDMulti, smpls)
    length(smpls) == m.n_rep ||
        throw(ArgumentError("smpls length $(length(smpls)) must equal n_rep=$(m.n_rep)"))
    n_u = length(m.all_unit_ids)
    for rep in smpls
        for fold in rep
            all(1 .<= fold.train .<= n_u) || throw(ArgumentError("train indices out of 1:n_units"))
            all(1 .<= fold.test .<= n_u) || throw(ArgumentError("test indices out of 1:n_units"))
        end
    end
    m.smpls = smpls
    m.use_unit_sample_splitting = true
    m.cell_smpls = nothing
    m.fitted = false
    return m
end

"""
    set_cell_sample_splitting!(m::DoubleMLDIDMulti, cell_smpls)

Set **per-ATT** sample splits. `cell_smpls` is a vector of length `n_att`, each
entry the usual `smpls` structure for that cell's cross-section (indices into
the cell's unit rows).
"""
function set_cell_sample_splitting!(m::DoubleMLDIDMulti, cell_smpls)
    length(cell_smpls) == length(m.gt_combos) ||
        throw(ArgumentError("cell_smpls length must equal n_att=$(length(m.gt_combos))"))
    m.cell_smpls = cell_smpls
    m.use_unit_sample_splitting = false
    m.fitted = false
    return m
end

"""
Project unit-level folds onto a cell subset (indices into `subset_uids`).

Units not in the subset are dropped from both train and test; train is
rebuilt as complement of test within the subset so every subset row appears
in exactly one test fold (empty folds are skipped / merged).
"""
function _project_unit_folds_to_subset(unit_smpls, all_unit_ids::Vector{Int},
                                       subset_uids::Vector{Int})
    n = length(subset_uids)
    row_of = Dict{Int,Int}(u => i for (i, u) in enumerate(subset_uids))
    mapped_reps = Vector{Any}(undef, length(unit_smpls))
    for (r, folds) in enumerate(unit_smpls)
        proj = NamedTuple{(:train, :test),Tuple{Vector{Int},Vector{Int}}}[]
        claimed = falses(n)
        for fold in folds
            te = Int[]
            for i in fold.test
                u = all_unit_ids[i]
                if haskey(row_of, u)
                    push!(te, row_of[u])
                    claimed[row_of[u]] = true
                end
            end
            isempty(te) && continue
            tr = setdiff(1:n, te)
            # require non-empty train when possible
            isempty(tr) && length(te) >= 2 && (tr = te[1:1]; te = te[2:end])
            isempty(tr) && continue
            push!(proj, (train=sort(tr), test=sort(te)))
        end
        # any unclaimed rows → append to last test or create fold
        unclaimed = findall(.!claimed)
        if !isempty(unclaimed)
            if isempty(proj)
                # fallback: single fold leave-one-out style not available — use K from n
                mapped_reps[r] = make_folds(n, min(3, n); shuffle_rows=false)
                continue
            end
            # put unclaimed into first fold's test, remove from its train
            f0 = proj[1]
            te = sort(unique(vcat(f0.test, unclaimed)))
            tr = setdiff(1:n, te)
            proj[1] = (train=sort(tr), test=te)
        end
        mapped_reps[r] = proj
    end
    return mapped_reps
end

"""
Build cross-section (ΔY, G, X) for one (g, t_pre, t_eval) with control group selection.
Returns `(data::DoubleMLData, unit_ids::Vector{Int})`.
"""
function _did_multi_cs(data::DoubleMLData, g::Int, t_pre::Int, t_eval::Int,
                       control_group::String, never::Real, anticipation::Int, t_values::Vector{Int})
    ids = data.id
    ts = data.t
    row_of = Dict{Tuple{Int,Int},Int}()
    for i in 1:length(ids)
        row_of[(Int(ids[i]), Int(ts[i]))] = i
    end
    ug = _unit_g_map(data)

    # Python not_yet_treated:
    #   max_g_value = t_values[min(where(t==eval_t) + anticipation, end)]
    # (based on eval period only, not max(g, eval))
    idx_eval = findfirst(==(t_eval), t_values)
    max_g = if idx_eval === nothing
        t_eval
    else
        t_values[min(idx_eval + anticipation, length(t_values))]
    end

    # Python keeps covariates from first of the two periods (min of pre/eval)
    t_x = min(t_pre, t_eval)

    Xrows = Vector{Vector{Float64}}()
    ydelta = Float64[]
    dgroup = Float64[]
    unit_ids = Int[]

    for u in sort(collect(keys(ug)))  # stable id order (Python sorts by id)
        g_u = ug[u]
        G_ind = isfinite(g_u) && isapprox(g_u, Float64(g); atol=0)
        if control_group == "never_treated"
            C_ind = _is_never_treated(g_u, never)
        else
            later = isfinite(g_u) && (g_u > max_g) && !G_ind
            C_ind = _is_never_treated(g_u, never) || later
        end
        (G_ind || C_ind) || continue
        G_ind && C_ind && continue  # disjoint
        haskey(row_of, (u, t_pre)) || continue
        haskey(row_of, (u, t_eval)) || continue
        haskey(row_of, (u, t_x)) || continue
        i_pre = row_of[(u, t_pre)]
        i_eval = row_of[(u, t_eval)]
        i_x = row_of[(u, t_x)]
        push!(ydelta, data.y[i_eval] - data.y[i_pre])
        push!(dgroup, G_ind ? 1.0 : 0.0)
        push!(Xrows, vec(data.x[i_x, :]))
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

    # all panel units for aligned IF space
    m.all_unit_ids = sort(unique(Int.(m.data.id)))
    n_u = length(m.all_unit_ids)
    id_index = Dict(u => i for (i, u) in enumerate(m.all_unit_ids))
    if_units = zeros(n_u, n_rep, n_att)

    # unit-level folds (shared across ATTs) when requested
    if m.use_unit_sample_splitting && isempty(m.smpls) && m.cell_smpls === nothing
        m.smpls = make_repeated_folds(n_u, m.n_folds, n_rep; rng=m.rng)
    end

    psi_list = Matrix{Float64}[]
    psi_d_list = Matrix{Float64}[]
    max_n = 0

    for (j, (g, t_pre, t_eval)) in enumerate(m.gt_combos)
        cs, uids = _did_multi_cs(
            m.data, g, t_pre, t_eval,
            m.control_group, m.never_treated_value, m.anticipation_periods, m.t_values,
        )
        cell_seed = UInt(hash((g, t_pre, t_eval, m.n_folds), UInt(0xD1D)))
        did = DoubleMLDID(
            cs, clone(m.ml_g), m.score == "observational" ? clone(m.ml_m) : nothing;
            n_folds=m.n_folds, n_rep=n_rep,
            score=m.score,
            in_sample_normalization=m.in_sample_normalization,
            trimming_threshold=m.trimming_threshold,
            ps_processor=m.ps_processor,
            draw_sample_splitting=false,
            rng=MersenneTwister(cell_seed),
        )
        # choose sample splits for this cell
        if m.cell_smpls !== nothing
            set_sample_splitting!(did, m.cell_smpls[j])
        elseif m.use_unit_sample_splitting && !isempty(m.smpls)
            mapped = _project_unit_folds_to_subset(m.smpls, m.all_unit_ids, uids)
            # mapped may have fewer folds; clamp n_folds for this cell
            if !isempty(mapped) && !isempty(mapped[1])
                did.n_folds = length(mapped[1])
                set_sample_splitting!(did, mapped)
            else
                # fallback independent folds
                did.smpls = make_repeated_folds(n_obs(cs), m.n_folds, n_rep;
                                                rng=MersenneTwister(cell_seed))
            end
        else
            did.smpls = make_repeated_folds(n_obs(cs), m.n_folds, n_rep;
                                            rng=MersenneTwister(cell_seed))
        end
        fit!(did; store_predictions=store_predictions)
        push!(m.att_models, did)
        push!(m.unit_ids_per_att, uids)
        all_coef[j, :] = did.all_coef[1, :]
        all_se[j, :] = did.all_se[1, :]
        max_n = max(max_n, size(did.psi, 1))
        push!(psi_list, did.psi[:, :, 1])
        push!(psi_d_list, did.psi_deriv[:, :, 1])

        # map IF onto full unit space: IF = ψ/J
        for r in 1:n_rep
            J = mean(@view did.psi_deriv[:, r, 1])
            abs(J) < 1e-14 && continue
            for (k, u) in enumerate(uids)
                iu = id_index[u]
                if_units[iu, r, j] = did.psi[k, r, 1] / J
            end
        end
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
    m.if_units = if_units
    m.boot = nothing
    m.fitted = true
    return m
end

"""
Multiplier bootstrap for multi-period DiD using **unit-aligned** influence functions
(so joint CIs / Romano–Wolf account for dependence across ATTs).
"""
function bootstrap!(m::DoubleMLDIDMulti; method::AbstractString="normal",
                    n_rep_boot::Int=500, rng::Union{Nothing,AbstractRNG}=nothing)
    m.fitted || error("Call fit! before bootstrap!")
    isempty(m.if_units) && error("Missing unit IFs; re-fit")
    rng = rng === nothing ? m.rng : rng
    n_u, n_rep, n_coef = size(m.if_units)
    boot_t = fill(NaN, n_rep_boot, n_coef, n_rep)
    for r in 1:n_rep
        W = _draw_weights(method, n_rep_boot, n_u, rng)  # n_boot × n_u
        for j in 1:n_coef
            IF = @view m.if_units[:, r, j]
            se_r = m.all_se[j, r]
            denom = n_u * se_r
            boot_t[:, j, r] = W * (IF ./ denom)
        end
    end
    m.boot = BootstrapResult(String(method), n_rep_boot, boot_t)
    return m
end

"""
    effects_table(m::DoubleMLDIDMulti; method=:eventstudy, post_only=false)

Aggregation table suitable for event-study / group / time plots
(Python `plot_effects` data layer).

Returns a `DataFrame` with columns `name`, `event_time` (or group/time key),
`coef`, `std_err`, `ci_lower`, `ci_upper`.
"""
function effects_table(m::DoubleMLDIDMulti; method::Symbol=:eventstudy,
                       post_only::Bool=false, level::Real=0.95)
    a = aggregate(m, method; post_only=post_only)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    # parse event time / keys from names like "e=-1", "g=2", "t=3"
    keys = Float64[]
    for nm in a.names
        mobj = match(r"=(-?\d+\.?\d*)", nm)
        push!(keys, mobj === nothing ? NaN : parse(Float64, mobj.captures[1]))
    end
    return DataFrame(
        name = a.names,
        key = keys,
        coef = a.coef,
        std_err = a.se,
        ci_lower = a.coef .- z .* a.se,
        ci_upper = a.coef .+ z .* a.se,
        method = fill(String(method), length(a.coef)),
        overall_coef = fill(a.overall_coef, length(a.coef)),
        overall_se = fill(a.overall_se, length(a.coef)),
    )
end

"""Alias for Python-oriented API (`plot_effects` returns data, not a figure)."""
plot_effects(m::DoubleMLDIDMulti; kwargs...) = effects_table(m; kwargs...)

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

"""
Result of [`aggregate`](@ref) for DID multi (Python `DoubleMLDIDAggregation`).

Fields align with Python naming where practical:
- `aggregation_method_name` / `aggregation_names` / `aggregation_weights`
- `overall_aggregation_weights` / overall point+SE
- `additional_information` — free-form metadata
"""
struct DIDAggregation
    method::String
    names::Vector{String}
    coef::Vector{Float64}
    se::Vector{Float64}
    weights_overall::Vector{Float64}  # weight of each aggregation in overall
    overall_coef::Float64
    overall_se::Float64
    aggregation_weights::Vector{Vector{Float64}}  # per-agg cell weights
    additional_information::Dict{String,Any}
end

# Python-compatible property aliases
aggregation_method_name(a::DIDAggregation) = a.method
aggregation_names(a::DIDAggregation) = a.names
overall_aggregation_weights(a::DIDAggregation) = a.weights_overall
n_aggregations(a::DIDAggregation) = length(a.coef)

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

"""Python `aggregated_summary` — same as [`summary_table`](@ref) without overall row."""
function aggregated_summary(a::DIDAggregation; level::Real=0.95)
    df = summary_table(a; level=level)
    return df[1:(end - 1), :]
end

"""Python `overall_summary` — single-row overall aggregation."""
function overall_summary(a::DIDAggregation; level::Real=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    t = a.overall_coef / a.overall_se
    return DataFrame(
        name = ["overall"],
        coef = [a.overall_coef],
        std_err = [a.overall_se],
        t = [t],
        pvalue = [2 * cdf(Normal(), -abs(t))],
        ci_lower = [a.overall_coef - z * a.overall_se],
        ci_upper = [a.overall_coef + z * a.overall_se],
    )
end

function confint(a::DIDAggregation; level::Real=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    return DataFrame(
        name = vcat(a.names, ["overall"]),
        lower = vcat(a.coef .- z .* a.se, [a.overall_coef - z * a.overall_se]),
        upper = vcat(a.coef .+ z .* a.se, [a.overall_coef + z * a.overall_se]),
        level = fill(Float64(level), length(a.coef) + 1),
    )
end

"""Plot-ready table for an aggregation (Python `plot_effects` data layer)."""
function plot_effects(a::DIDAggregation; level::Real=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    keys = Float64[]
    for nm in a.names
        mobj = match(r"=(-?\d+\.?\d*)", nm)
        push!(keys, mobj === nothing ? NaN : parse(Float64, mobj.captures[1]))
    end
    return DataFrame(
        name = a.names,
        key = keys,
        coef = a.coef,
        std_err = a.se,
        ci_lower = a.coef .- z .* a.se,
        ci_upper = a.coef .+ z .* a.se,
        method = fill(a.method, length(a.names)),
    )
end

"""Share of units in each treatment group (balanced panel unit-level)."""
function _group_shares(data::DoubleMLData, g_values::Vector{Int}, never::Real)
    ug = _unit_g_map(data)
    n = length(ug)
    shares = Dict{Int,Float64}()
    for g in g_values
        _is_never_treated(g, never) && continue
        shares[g] = count(v -> isfinite(v) && isapprox(v, g; atol=0), values(ug)) / n
    end
    return shares
end

function _weighted_att(m::DoubleMLDIDMulti, idx::Vector{Int}, w::Vector{Float64})
    # coef = Σ w_j θ_j using unit-aligned influence functions for joint SE
    c = sum(w[k] * m.coef[idx[k]] for k in eachindex(idx))
    n_u, n_rep, _ = size(m.if_units)
    # average IF across reps then se
    if_agg = zeros(n_u)
    for r in 1:n_rep
        v = zeros(n_u)
        for k in eachindex(idx)
            v .+= w[k] .* @view(m.if_units[:, r, idx[k]])
        end
        if_agg .+= v
    end
    if_agg ./= n_rep
    se = sqrt(mean(if_agg .^ 2) / n_u)
    return c, se
end

function _combine_aggs(coefs, ses, w)
    c = sum(w .* coefs)
    # if we only have diagonal ses, still use weighted quadratic form
    v = sum((w .* ses) .^ 2)
    return c, sqrt(max(v, 0.0))
end

function _combine_aggs_if(m::DoubleMLDIDMulti, idx_list::Vector{Vector{Int}},
                          w_list::Vector{Vector{Float64}}, w_ov::Vector{Float64})
    # overall = sum_a w_ov[a] * sum_k w_list[a][k] * θ_{idx_list[a][k]}
    n_u, n_rep, _ = size(m.if_units)
    c = 0.0
    if_ov = zeros(n_u)
    for (a, idx) in enumerate(idx_list)
        wa = w_list[a]
        c += w_ov[a] * sum(wa[k] * m.coef[idx[k]] for k in eachindex(idx))
        for r in 1:n_rep
            v = zeros(n_u)
            for k in eachindex(idx)
                v .+= wa[k] .* @view(m.if_units[:, r, idx[k]])
            end
            if_ov .+= (w_ov[a] / n_rep) .* v
        end
    end
    se = sqrt(mean(if_ov .^ 2) / n_u)
    return c, se
end

"""
    aggregate(m::DoubleMLDIDMulti, method=:group; post_only=true) -> DIDAggregation

Callaway–Sant'Anna aggregations with **unit-aligned influence-function SEs**:

| `method` | Meaning |
|----------|---------|
| `:group` | Average ATT within each first-treatment group `g` |
| `:time` | Average ATT at each calendar time `t_eval` |
| `:eventstudy` | Average ATT by event time `e = t_eval − g` |
"""
function aggregate(m::DoubleMLDIDMulti, method::Symbol=:group; post_only::Bool=true)
    m.fitted || error("Call fit! before aggregate")
    method in (:group, :time, :eventstudy) ||
        throw(ArgumentError("method must be :group, :time, or :eventstudy"))
    isempty(m.if_units) && error("Model missing unit IFs; re-fit with updated package")

    n_att = length(m.gt_combos)
    gvec = [c[1] for c in m.gt_combos]
    teval = [c[3] for c in m.gt_combos]
    evec = teval .- gvec
    shares = _group_shares(m.data, m.g_values, m.never_treated_value)
    sel = method === :eventstudy ? trues(n_att) : (post_only ? (evec .>= 0) : trues(n_att))
    any(sel) || error("No ATT cells selected for aggregation")

    names = String[]; coefs = Float64[]; ses = Float64[]
    idx_list = Vector{Int}[]; w_list = Vector{Float64}[]; w_ov = Float64[]

    if method === :group
        groups = sort(unique(gvec[sel]))
        for g in groups
            idx = findall(i -> sel[i] && gvec[i] == g, 1:n_att)
            w = fill(1 / length(idx), length(idx))
            c, s = _weighted_att(m, idx, w)
            push!(names, "g=$g"); push!(coefs, c); push!(ses, s)
            push!(idx_list, idx); push!(w_list, w)
            push!(w_ov, get(shares, g, 0.0))
        end
        sw = sum(w_ov); sw > 0 || (w_ov .= 1 / length(w_ov); sw = 1.0); w_ov ./= sw
    elseif method === :time
        times = sort(unique(teval[sel]))
        for t in times
            idx = findall(i -> sel[i] && teval[i] == t, 1:n_att)
            ws = Float64[get(shares, gvec[i], 0.0) for i in idx]
            ssum = sum(ws); ssum > 0 || (ws .= 1 / length(ws); ssum = 1.0); ws ./= ssum
            c, s = _weighted_att(m, idx, ws)
            push!(names, "t=$t"); push!(coefs, c); push!(ses, s)
            push!(idx_list, idx); push!(w_list, ws)
        end
        w_ov = fill(1 / length(coefs), length(coefs))
    else
        es = sort(unique(evec[sel]))
        n_post = count(>=(0), es)
        for e in es
            idx = findall(i -> sel[i] && evec[i] == e, 1:n_att)
            ws = Float64[get(shares, gvec[i], 0.0) for i in idx]
            ssum = sum(ws); ssum > 0 || (ws .= 1 / length(ws); ssum = 1.0); ws ./= ssum
            c, s = _weighted_att(m, idx, ws)
            push!(names, "e=$e"); push!(coefs, c); push!(ses, s)
            push!(idx_list, idx); push!(w_list, ws)
            push!(w_ov, e >= 0 ? 1 / max(n_post, 1) : 0.0)
        end
        if sum(w_ov) <= 0
            w_ov .= 1 / length(w_ov)
        else
            w_ov ./= sum(w_ov)
        end
    end

    oc, os = _combine_aggs_if(m, idx_list, w_list, w_ov)
    info = Dict{String,Any}(
        "Score function" => m.score,
        "Control group" => m.control_group,
        "Anticipation periods" => m.anticipation_periods,
        "post_only" => post_only,
    )
    return DIDAggregation(String(method), names, coefs, ses, w_ov, oc, os, w_list, info)
end

"""
    p_adjust(m::DoubleMLDIDMulti; method=:romano_wolf) -> DataFrame

Multiple-testing adjusted p-values for group–time ATTs.
`method`: `:romano_wolf` (bootstrap stepdown), `:bonferroni`, `:holm`.
Requires [`bootstrap!`](@ref) for Romano–Wolf.
"""
function p_adjust(m::DoubleMLDIDMulti; method::Symbol=:romano_wolf, level::Real=0.95)
    m.fitted || error("Call fit! first")
    raw = pval(m)
    names = m.treat_names
    if method === :bonferroni
        adj = min.(1.0, raw .* length(raw))
    elseif method === :holm
        adj = _holm_adjust(raw)
    elseif method === :romano_wolf
        m.boot === nothing && error("Apply bootstrap! before p_adjust(:romano_wolf)")
        adj = _romano_wolf_adjust(m)
    else
        throw(ArgumentError("method must be :romano_wolf, :bonferroni, or :holm"))
    end
    return DataFrame(att=names, pvalue=raw, pvalue_adjusted=adj, method=fill(String(method), length(raw)))
end

function _holm_adjust(p::Vector{Float64})
    n = length(p)
    ord = sortperm(p)
    adj = similar(p)
    for (rank, i) in enumerate(ord)
        adj[i] = min(1.0, p[i] * (n - rank + 1))
    end
    # enforce monotonicity
    for k in 2:n
        adj[ord[k]] = max(adj[ord[k]], adj[ord[k-1]])
    end
    return adj
end

function _romano_wolf_adjust(m::DoubleMLDIDMulti)
    # stepdown using bootstrap t-stats
    t0 = abs.(t_stat(m))
    boot = abs.(m.boot.boot_t_stat)  # n_boot × n_coef × n_rep
    n_boot, n_coef, n_rep = size(boot)
    # median over reps
    boot_med = mapslices(median, boot; dims=3)[:, :, 1]  # n_boot × n_coef
    ord = sortperm(t0; rev=true)  # largest |t| first
    adj = ones(n_coef)
    for (step, j) in enumerate(ord)
        # max |t| over remaining hypotheses
        remain = ord[step:end]
        max_boot = maximum(boot_med[:, remain]; dims=2)[:]
        adj[j] = mean(max_boot .>= t0[j])
    end
    # enforce increasing in stepdown order
    for step in 2:n_coef
        adj[ord[step]] = max(adj[ord[step]], adj[ord[step-1]])
    end
    return adj
end
