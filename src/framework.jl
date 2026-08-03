# DoubleMLFramework — combine estimates via influence functions
# Python: doubleml.DoubleMLFramework / DoubleMLCore / concat
#
# Model psi layout:  (n_obs, n_rep, n_coef)
# Framework scaled_psi: (n_obs, n_thetas, n_rep)  — Python layout for concat on θ-axis

"""
    DoubleMLCore

Internal container for raw DML estimates and scaled scores.
"""
struct DoubleMLCore
    all_thetas::Matrix{Float64}          # n_thetas × n_rep
    all_ses::Matrix{Float64}             # n_thetas × n_rep
    var_scaling_factors::Vector{Float64} # length n_thetas
    scaled_psi::Array{Float64,3}         # n_obs × n_thetas × n_rep
    is_cluster_data::Bool
    cluster_dict::Union{Nothing,NamedTuple}
end

"""
    DoubleMLFramework

Combine and infer on DML estimates (Python `DoubleMLFramework`).

Construct from a fitted model via [`construct_framework`](@ref), or combine
via `+`, `-`, `*`, and [`concat`](@ref).
"""
mutable struct DoubleMLFramework
    core::DoubleMLCore
    treatment_names::Vector{String}
    thetas::Vector{Float64}
    ses::Vector{Float64}
    boot::Union{Nothing,BootstrapResult}
end

function DoubleMLFramework(core::DoubleMLCore;
                           treatment_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing)
    n_θ = size(core.all_thetas, 1)
    names = if treatment_names === nothing
        ["theta_$i" for i in 1:n_θ]
    else
        length(treatment_names) == n_θ ||
            throw(ArgumentError("treatment_names length must equal n_thetas=$n_θ"))
        String.(collect(treatment_names))
    end
    thetas, ses = aggregate_reps(core.all_thetas, core.all_ses)
    return DoubleMLFramework(core, names, thetas, ses, nothing)
end

# ---- construct from fitted model -------------------------------------------

"""
    construct_framework(m::AbstractDoubleML) -> DoubleMLFramework

Build a framework from a fitted model's scores:

```
scaled_psi[:, j, r] = psi[:, r, j] / mean(psi_deriv[:, r, j])
```
"""
function construct_framework(m::AbstractDoubleML)
    m.fitted || error("Call fit! before construct_framework")
    hasproperty(m, :psi) || error("Model does not store psi")
    hasproperty(m, :psi_deriv) || error("Model does not store psi_deriv")

    n, n_rep, n_coef = size(m.psi)
    scaled = Array{Float64,3}(undef, n, n_coef, n_rep)
    for r in 1:n_rep
        for j in 1:n_coef
            J = mean(@view m.psi_deriv[:, r, j])
            abs(J) < 1e-14 && error("Degenerate psi_deriv for coef $j, rep $r")
            scaled[:, j, r] = @view(m.psi[:, r, j]) ./ J
        end
    end

    vsf = _var_scaling_factors(m)
    is_cl = hasproperty(m, :is_cluster_data) && m.is_cluster_data === true
    cdict = if hasproperty(m, :cluster_dict) && m.cluster_dict !== nothing
        m.cluster_dict
    else
        nothing
    end

    core = DoubleMLCore(
        copy(m.all_coef), copy(m.all_se), vsf, scaled, is_cl, cdict,
    )
    names = hasproperty(m, :treat_names) ? m.treat_names : ["theta_$i" for i in 1:n_coef]
    return DoubleMLFramework(core; treatment_names=names)
end

"""Default var scaling = n_obs for each coefficient (iid)."""
function _var_scaling_factors(m::AbstractDoubleML)
    n = size(m.psi, 1)
    n_coef = size(m.psi, 3)
    if hasproperty(m, :var_scaling) && m.var_scaling !== nothing &&
       length(m.var_scaling) == n_coef
        return Float64.(m.var_scaling)
    end
    return fill(Float64(n), n_coef)
end

"""Alias: `framework(m)` → `construct_framework(m)`."""
framework(m::AbstractDoubleML) = construct_framework(m)

# ---- rebuild SE from scaled_psi --------------------------------------------

function _rebuild_all_ses(scaled_psi::Array{Float64,3},
                          var_scaling::AbstractVector{<:Real})
    n_obs, n_θ, n_rep = size(scaled_psi)
    all_ses = zeros(n_θ, n_rep)
    for r in 1:n_rep
        for j in 1:n_θ
            s2 = mean((@view scaled_psi[:, j, r]) .^ 2)
            all_ses[j, r] = sqrt(s2 / var_scaling[j])
        end
    end
    return all_ses
end

function _check_compatible_add(a::DoubleMLFramework, b::DoubleMLFramework)
    ca, cb = a.core, b.core
    size(ca.scaled_psi) == size(cb.scaled_psi) ||
        throw(DimensionMismatch("scaled_psi shapes differ: $(size(ca.scaled_psi)) vs $(size(cb.scaled_psi))"))
    ca.var_scaling_factors ≈ cb.var_scaling_factors ||
        throw(ArgumentError("var_scaling_factors must match for +/-"))
    ca.is_cluster_data == cb.is_cluster_data ||
        throw(ArgumentError("cluster flags must match for +/-"))
    return nothing
end

# ---- arithmetic ------------------------------------------------------------

function Base.:+(a::DoubleMLFramework, b::DoubleMLFramework)
    _check_compatible_add(a, b)
    ca, cb = a.core, b.core
    all_θ = ca.all_thetas .+ cb.all_thetas
    scaled = ca.scaled_psi .+ cb.scaled_psi
    all_se = _rebuild_all_ses(scaled, ca.var_scaling_factors)
    core = DoubleMLCore(all_θ, all_se, copy(ca.var_scaling_factors), scaled,
                        ca.is_cluster_data, ca.cluster_dict)
    names = ["($(a.treatment_names[i])+$(b.treatment_names[i]))" for i in eachindex(a.treatment_names)]
    return DoubleMLFramework(core; treatment_names=names)
end

function Base.:-(a::DoubleMLFramework, b::DoubleMLFramework)
    _check_compatible_add(a, b)
    ca, cb = a.core, b.core
    all_θ = ca.all_thetas .- cb.all_thetas
    scaled = ca.scaled_psi .- cb.scaled_psi
    all_se = _rebuild_all_ses(scaled, ca.var_scaling_factors)
    core = DoubleMLCore(all_θ, all_se, copy(ca.var_scaling_factors), scaled,
                        ca.is_cluster_data, ca.cluster_dict)
    names = ["($(a.treatment_names[i])-$(b.treatment_names[i]))" for i in eachindex(a.treatment_names)]
    return DoubleMLFramework(core; treatment_names=names)
end

function Base.:*(c::Real, a::DoubleMLFramework)
    ca = a.core
    all_θ = c .* ca.all_thetas
    scaled = c .* ca.scaled_psi
    all_se = abs(c) .* ca.all_ses
    core = DoubleMLCore(all_θ, all_se, copy(ca.var_scaling_factors), scaled,
                        ca.is_cluster_data, ca.cluster_dict)
    names = ["$(c)*$(nm)" for nm in a.treatment_names]
    return DoubleMLFramework(core; treatment_names=names)
end
Base.:*(a::DoubleMLFramework, c::Real) = c * a

"""
    concat(fs) -> DoubleMLFramework

Stack frameworks along the parameter axis (Python `doubleml.concat`).
All must share `n_obs` and `n_rep`.
"""
function concat(fs::AbstractVector{<:DoubleMLFramework})
    length(fs) >= 1 || throw(ArgumentError("concat needs at least one framework"))
    n_obs = size(fs[1].core.scaled_psi, 1)
    n_rep = size(fs[1].core.scaled_psi, 3)
    for f in fs
        size(f.core.scaled_psi, 1) == n_obs || throw(DimensionMismatch("n_obs mismatch in concat"))
        size(f.core.scaled_psi, 3) == n_rep || throw(DimensionMismatch("n_rep mismatch in concat"))
    end
    all_θ = vcat((f.core.all_thetas for f in fs)...)
    all_se = vcat((f.core.all_ses for f in fs)...)
    vsf = vcat((f.core.var_scaling_factors for f in fs)...)
    scaled = cat((f.core.scaled_psi for f in fs)...; dims=2)
    is_cl = any(f.core.is_cluster_data for f in fs)
    cdict = fs[1].core.cluster_dict
    core = DoubleMLCore(all_θ, all_se, vsf, scaled, is_cl, cdict)
    names = vcat((f.treatment_names for f in fs)...)
    return DoubleMLFramework(core; treatment_names=names)
end
concat(fs::DoubleMLFramework...) = concat(collect(fs))

# ---- inference -------------------------------------------------------------

function t_stat(f::DoubleMLFramework)
    return f.thetas ./ f.ses
end

function pval(f::DoubleMLFramework)
    return 2 .* cdf.(Normal(), -abs.(t_stat(f)))
end

"""
    bootstrap!(f::DoubleMLFramework; method="normal", n_rep_boot=500, rng=...)

Multiplier bootstrap on framework scaled scores.
"""
function bootstrap!(f::DoubleMLFramework; method::AbstractString="normal",
                    n_rep_boot::Int=500, rng::AbstractRNG=Random.default_rng())
    n_rep_boot >= 1 || throw(ArgumentError("n_rep_boot must be ≥ 1"))
    n_obs, n_θ, n_rep = size(f.core.scaled_psi)
    boot_t = fill(NaN, n_rep_boot, n_θ, n_rep)
    for r in 1:n_rep
        W = _draw_weights(method, n_rep_boot, n_obs, rng)
        for j in 1:n_θ
            se_r = f.core.all_ses[j, r]
            denom = f.core.var_scaling_factors[j] * se_r
            boot_t[:, j, r] = W * (@view(f.core.scaled_psi[:, j, r]) ./ denom)
        end
    end
    f.boot = BootstrapResult(String(method), n_rep_boot, boot_t)
    return f
end

function confint(f::DoubleMLFramework; level::Real=0.95, joint::Bool=false)
    (0 < level < 1) || throw(ArgumentError("level must be in (0,1)"))
    n_θ = length(f.thetas)
    n_rep = size(f.core.all_thetas, 2)
    if joint
        f.boot === nothing && error("Apply bootstrap! before confint(joint=true)")
        max_abs = maximum(abs.(f.boot.boot_t_stat); dims=2)
        crit = [quantile(vec(max_abs[:, 1, r]), level) for r in 1:n_rep]
    else
        z = quantile(Normal(), 1 - (1 - level) / 2)
        crit = fill(z, n_rep)
    end
    lower_reps = f.core.all_thetas .- f.core.all_ses .* reshape(crit, 1, n_rep)
    upper_reps = f.core.all_thetas .+ f.core.all_ses .* reshape(crit, 1, n_rep)
    lower = vec(median(lower_reps; dims=2))
    upper = vec(median(upper_reps; dims=2))
    return DataFrame(
        treatment = f.treatment_names,
        lower = lower,
        upper = upper,
        level = fill(Float64(level), n_θ),
        joint = fill(joint, n_θ),
    )
end

function summary_table(f::DoubleMLFramework; level::Real=0.95)
    ci = confint(f; level=level)
    return DataFrame(
        treatment = f.treatment_names,
        coef = f.thetas,
        std_err = f.ses,
        t = t_stat(f),
        pvalue = pval(f),
        ci_lower = ci.lower,
        ci_upper = ci.upper,
    )
end

function p_adjust(f::DoubleMLFramework; method::Symbol=:holm)
    # reuse AbstractDoubleML logic via a thin wrapper
    raw = pval(f)
    n = length(raw)
    if method === :bonferroni
        adj = min.(1.0, raw .* n)
    elseif method === :holm
        ord = sortperm(raw)
        adj = similar(raw)
        for (rank, i) in enumerate(ord)
            adj[i] = min(1.0, raw[i] * (n - rank + 1))
        end
        for k in 2:n
            adj[ord[k]] = max(adj[ord[k]], adj[ord[k - 1]])
        end
    elseif method === :romano_wolf
        f.boot === nothing && error("Apply bootstrap! before p_adjust(:romano_wolf)")
        t0 = abs.(t_stat(f))
        boot = abs.(f.boot.boot_t_stat)
        boot_med = mapslices(median, boot; dims=3)[:, :, 1]
        ord = sortperm(t0; rev=true)
        adj = ones(n)
        for (step, j) in enumerate(ord)
            remain = ord[step:end]
            max_boot = vec(maximum(boot_med[:, remain]; dims=2))
            adj[j] = mean(max_boot .>= t0[j])
        end
        for step in 2:n
            adj[ord[step]] = max(adj[ord[step]], adj[ord[step - 1]])
        end
    else
        throw(ArgumentError("method must be :holm, :bonferroni, or :romano_wolf"))
    end
    return DataFrame(parameter=f.treatment_names, pvalue=raw, pvalue_adjusted=adj,
                     method=fill(String(method), n))
end

function Base.show(io::IO, f::DoubleMLFramework)
    print(io, "DoubleMLFramework(n_thetas=$(length(f.thetas)), ",
          "thetas=$(round.(f.thetas; digits=4)), ses=$(round.(f.ses; digits=4)))")
end
