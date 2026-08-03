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
    # optional OVB sensitivity (copied from PLR/IRM via construct_framework)
    # length == n_thetas when present
    sens_elements::Union{Nothing,Vector{SensitivityElements}}
    sensitivity::Union{Nothing,SensitivityResult}
end

function DoubleMLFramework(core::DoubleMLCore;
                           treatment_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
                           sens_elements::Union{Nothing,Vector{SensitivityElements}}=nothing)
    n_θ = size(core.all_thetas, 1)
    names = if treatment_names === nothing
        ["theta_$i" for i in 1:n_θ]
    else
        length(treatment_names) == n_θ ||
            throw(ArgumentError("treatment_names length must equal n_thetas=$n_θ"))
        String.(collect(treatment_names))
    end
    thetas, ses = aggregate_reps(core.all_thetas, core.all_ses)
    return DoubleMLFramework(core, names, thetas, ses, nothing, sens_elements, nothing)
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
    # carry sensitivity building blocks (one SensitivityElements per θ)
    selem = if hasproperty(m, :sens_elements) && m.sens_elements !== nothing
        els = m.sens_elements isa Vector ? m.sens_elements : SensitivityElements[m.sens_elements]
        length(els) == n_coef ? els : nothing  # drop if length mismatch
    else
        nothing
    end
    return DoubleMLFramework(core; treatment_names=names, sens_elements=selem)
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
    # arithmetic drops OVB elements (not generally valid under +/−)
    return DoubleMLFramework(core; treatment_names=names, sens_elements=nothing)
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
    return DoubleMLFramework(core; treatment_names=names, sens_elements=nothing)
end

function Base.:*(c::Real, a::DoubleMLFramework)
    ca = a.core
    all_θ = c .* ca.all_thetas
    scaled = c .* ca.scaled_psi
    all_se = abs(c) .* ca.all_ses
    core = DoubleMLCore(all_θ, all_se, copy(ca.var_scaling_factors), scaled,
                        ca.is_cluster_data, ca.cluster_dict)
    names = ["$(c)*$(nm)" for nm in a.treatment_names]
    # scaling preserves elements only for c == ±1; drop otherwise
    selem = (abs(abs(c) - 1) < 1e-14) ? a.sens_elements : nothing
    return DoubleMLFramework(core; treatment_names=names, sens_elements=selem)
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

"""Python-compatible alias for framework contour grid."""
sensitivity_plot(f::DoubleMLFramework; kwargs...) = sensitivity_contour(f; kwargs...)

# ---- Framework sensitivity (Python DoubleMLFramework.sensitivity_*) --------

"""
    sensitivity_analysis!(f::DoubleMLFramework; cf_y=0.03, cf_d=0.03, rho=1.0,
                          level=0.95, null_hypothesis=0.0)

OVB sensitivity on a framework built from a model that stored sensitivity
elements (PLR / IRM via [`construct_framework`](@ref)). Supports multi-θ.
"""
function sensitivity_analysis!(f::DoubleMLFramework;
                               cf_y::Real=0.03,
                               cf_d::Real=0.03,
                               rho::Real=1.0,
                               level::Real=0.95,
                               null_hypothesis::Union{Real,AbstractVector{<:Real}}=0.0)
    f.sens_elements === nothing &&
        error("Framework has no sensitivity elements — construct from fitted PLR/IRM")
    els = f.sens_elements
    n_coef = length(f.thetas)
    length(els) == n_coef ||
        error("sens_elements length $(length(els)) ≠ n_thetas=$n_coef")
    nulls = null_hypothesis isa AbstractVector ?
        Float64.(collect(null_hypothesis)) :
        fill(Float64(null_hypothesis), n_coef)
    length(nulls) == n_coef || throw(ArgumentError("null_hypothesis length must equal n_thetas"))
    strength = confounding_strength(cf_y, cf_d, rho)

    θ_lo = zeros(n_coef); θ_hi = zeros(n_coef)
    σ_lo = zeros(n_coef); σ_hi = zeros(n_coef)
    ci_lo = zeros(n_coef); ci_hi = zeros(n_coef)
    rvs = zeros(n_coef); rvas = zeros(n_coef)

    for j in 1:n_coef
        selem = els[j]
        n_rep = length(selem.sigma2)
        all_psi_scaled = similar(selem.psi_sigma2)
        all_psi_max = similar(selem.psi_sigma2)
        all_max_bias = zeros(n_rep)
        all_theta = vec(f.core.all_thetas[j, :])
        for r in 1:n_rep
            mb, pmb = _max_bias_and_if(selem.sigma2[r], selem.nu2[r],
                                       @view(selem.psi_sigma2[:, r]), @view(selem.psi_nu2[:, r]))
            all_max_bias[r] = mb
            all_psi_max[:, r] = pmb
            all_psi_scaled[:, r] = @view f.core.scaled_psi[:, j, r]
        end
        bounds = _sensitivity_bounds(all_theta, all_max_bias, all_psi_scaled, all_psi_max, strength, level)
        function calc_c(c)
            s = confounding_strength(c, c, rho)
            _sensitivity_bounds(all_theta, all_max_bias, all_psi_scaled, all_psi_max, s, level)
        end
        θ̂ = f.thetas[j]
        θ_lo[j] = bounds.theta_lower; θ_hi[j] = bounds.theta_upper
        σ_lo[j] = bounds.se_lower; σ_hi[j] = bounds.se_upper
        ci_lo[j] = bounds.ci_lower; ci_hi[j] = bounds.ci_upper
        rvs[j] = _robustness_value(calc_c, nulls[j], θ̂; which=:theta)
        rvas[j] = _robustness_value(calc_c, nulls[j], θ̂; which=:ci)
    end

    result = SensitivityResult(
        Float64(cf_y), Float64(cf_d), Float64(rho), Float64(level), nulls[1],
        θ_lo, θ_hi, σ_lo, σ_hi, ci_lo, ci_hi, rvs, rvas,
    )
    f.sensitivity = result
    return result
end

function sensitivity_summary(f::DoubleMLFramework)
    r = f.sensitivity
    r === nothing && return "Apply sensitivity_analysis! first."
    io = IOBuffer()
    println(io, "================== Sensitivity Analysis ==================")
    println(io, "------------------ Scenario          ------------------")
    println(io, "Significance Level: level=$(r.level)")
    println(io, "Sensitivity parameters: cf_y=$(r.cf_y); cf_d=$(r.cf_d), rho=$(r.rho)")
    println(io, "------------------ Bounds with CI    ------------------")
    println(io, DataFrame(
        treatment = f.treatment_names,
        ci_lower = r.ci_lower,
        theta_lower = r.theta_lower,
        theta = f.thetas,
        theta_upper = r.theta_upper,
        ci_upper = r.ci_upper,
    ))
    println(io, "------------------ Robustness Values ------------------")
    println(io, DataFrame(treatment=f.treatment_names, rv=r.rv, rva=r.rva,
                          null_hypothesis=fill(r.null_hypothesis, length(r.rv))))
    return String(take!(io))
end

"""
    sensitivity_contour(f::DoubleMLFramework; idx_treatment=1, ...) -> DataFrame

Contour grid for one treatment index of a framework with sensitivity elements.
"""
function sensitivity_contour(f::DoubleMLFramework;
                             cf_y_max::Real=0.15,
                             cf_d_max::Real=0.15,
                             grid_size::Int=20,
                             rho::Real=1.0,
                             level::Real=0.95,
                             null_hypothesis::Real=0.0,
                             value::Symbol=:theta,
                             idx_treatment::Int=1)
    f.sens_elements === nothing &&
        error("Framework has no sensitivity elements")
    value in (:theta, :ci) || throw(ArgumentError("value must be :theta or :ci"))
    grid_size >= 2 || throw(ArgumentError("grid_size ≥ 2"))
    1 <= idx_treatment <= length(f.sens_elements) ||
        throw(ArgumentError("idx_treatment out of range"))
    j = idx_treatment
    selem = f.sens_elements[j]
    n_rep = length(selem.sigma2)
    all_psi_scaled = similar(selem.psi_sigma2)
    all_psi_max = similar(selem.psi_sigma2)
    all_max_bias = zeros(n_rep)
    all_theta = vec(f.core.all_thetas[j, :])
    for r in 1:n_rep
        mb, pmb = _max_bias_and_if(selem.sigma2[r], selem.nu2[r],
                                   @view(selem.psi_sigma2[:, r]), @view(selem.psi_nu2[:, r]))
        all_max_bias[r] = mb
        all_psi_max[:, r] = pmb
        all_psi_scaled[:, r] = @view f.core.scaled_psi[:, j, r]
    end

    ys = range(0.0, Float64(cf_y_max); length=grid_size)
    ds = range(0.0, Float64(cf_d_max); length=grid_size)
    rows_cfy = Float64[]; rows_cfd = Float64[]
    tl = Float64[]; tu = Float64[]; cl = Float64[]; cu = Float64[]
    covers = Bool[]
    for cy in ys, cd in ds
        strength = confounding_strength(cy, cd, rho)
        b = _sensitivity_bounds(all_theta, all_max_bias, all_psi_scaled, all_psi_max, strength, level)
        push!(rows_cfy, cy); push!(rows_cfd, cd)
        push!(tl, b.theta_lower); push!(tu, b.theta_upper)
        push!(cl, b.ci_lower); push!(cu, b.ci_upper)
        if value === :theta
            push!(covers, b.theta_lower <= null_hypothesis <= b.theta_upper)
        else
            push!(covers, b.ci_lower <= null_hypothesis <= b.ci_upper)
        end
    end
    return DataFrame(
        cf_y = rows_cfy, cf_d = rows_cfd,
        theta_lower = tl, theta_upper = tu,
        ci_lower = cl, ci_upper = cu,
        covers_null = covers,
        rho = fill(Float64(rho), length(rows_cfy)),
        level = fill(Float64(level), length(rows_cfy)),
        value = fill(String(value), length(rows_cfy)),
        treatment = fill(f.treatment_names[j], length(rows_cfy)),
    )
end
