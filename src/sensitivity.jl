# Sensitivity analysis for omitted variable bias
# (Chernozhukov et al. 2022 / Python DoubleML.sensitivity_analysis)

"""
    SensitivityElements

Raw building blocks stored at `fit!` time for PLR / IRM.
Shapes: scalars σ², ν² and vectors of length n for influence-function pieces
(per coefficient / rep; we store for the single-treatment case as vectors of length n × n_rep).
"""
struct SensitivityElements
    # per rep: length n_rep
    sigma2::Vector{Float64}
    nu2::Vector{Float64}
    # n × n_rep
    psi_sigma2::Matrix{Float64}
    psi_nu2::Matrix{Float64}
    riesz_rep::Matrix{Float64}
end

"""
    SensitivityResult

Result of [`sensitivity_analysis!`](@ref) for a fixed confounding scenario.
"""
struct SensitivityResult
    cf_y::Float64
    cf_d::Float64
    rho::Float64
    level::Float64
    null_hypothesis::Float64
    # bounds on θ
    theta_lower::Vector{Float64}
    theta_upper::Vector{Float64}
    se_lower::Vector{Float64}
    se_upper::Vector{Float64}
    ci_lower::Vector{Float64}
    ci_upper::Vector{Float64}
    # robustness values (cf_y = cf_d = rv that just covers null)
    rv::Vector{Float64}
    rva::Vector{Float64}
end

function Base.show(io::IO, r::SensitivityResult)
    print(io,
        "SensitivityResult(cf_y=$(r.cf_y), cf_d=$(r.cf_d), rho=$(r.rho); ",
        "θ∈[$(round(r.theta_lower[1]; digits=4)), $(round(r.theta_upper[1]; digits=4))], ",
        "rv=$(round(r.rv[1]; digits=4)), rva=$(round(r.rva[1]; digits=4)))")
end

# ---- building blocks from residuals (PLR) ----

"""
PLR sensitivity elements (Python `DoubleMLPLR._sensitivity_element_est`).

For score `"partialling out"`:
`σ² = E[(Y − ℓ̂ − θ(D − m̂))²]`, `ν² = 1 / E[(D − m̂)²]`,
Riesz representer `α = (D − m̂) ν²`.
"""
function sensitivity_elements_plr(y, d, l_hat, m_hat, θ; score::AbstractString="partialling out")
    n = length(y)
    if score == "partialling out"
        resid = y .- l_hat .- θ .* (d .- m_hat)
    else
        # IV-type: needs g_hat; caller passes l_hat slot as g_hat
        resid = y .- l_hat .- θ .* d
    end
    sigma2_i = resid .^ 2
    sigma2 = mean(sigma2_i)
    psi_sigma2 = sigma2_i .- sigma2

    v = d .- m_hat
    nu2 = 1.0 / mean(v .^ 2)
    psi_nu2 = nu2 .- (v .^ 2) .* (nu2^2)
    rr = v .* nu2

    return sigma2, nu2, psi_sigma2, psi_nu2, rr
end

"""
IRM ATE sensitivity elements (Python `DoubleMLIRM._sensitivity_element_est` with unit weights).
"""
function sensitivity_elements_irm_ate(y, d, g0, g1, m_hat)
    m_hat = clamp.(m_hat, 1e-6, 1 - 1e-6)
    resid = y .- d .* g1 .- (1 .- d) .* g0
    sigma2_i = resid .^ 2
    sigma2 = mean(sigma2_i)
    psi_sigma2 = sigma2_i .- sigma2

    # ATE: weights = weights_bar = 1
    m_alpha = (1 ./ m_hat) .+ (1 ./ (1 .- m_hat))
    rr = d ./ m_hat .- (1 .- d) ./ (1 .- m_hat)
    nu2_i = 2 .* m_alpha .- rr .^ 2
    nu2 = mean(nu2_i)
    # guard against numerical negativity
    if nu2 <= 0
        nu2 = max(mean(rr .^ 2), eps())
        nu2_i = fill(nu2, length(y))
    end
    psi_nu2 = nu2_i .- nu2
    return sigma2, nu2, psi_sigma2, psi_nu2, rr
end

function _max_bias_and_if(sigma2, nu2, psi_sigma2, psi_nu2)
    max_bias = sqrt(max(sigma2, 0.0) * max(nu2, 0.0))
    if max_bias < 1e-14
        psi_max = zeros(length(psi_sigma2))
    else
        psi_max = (sigma2 .* psi_nu2 .+ nu2 .* psi_sigma2) ./ (2 * max_bias)
    end
    return max_bias, psi_max
end

function confounding_strength(cf_y::Real, cf_d::Real, rho::Real)
    (0 <= cf_y < 1) || throw(ArgumentError("cf_y must be in [0,1)"))
    (0 <= cf_d < 1) || throw(ArgumentError("cf_d must be in [0,1)"))
    abs(rho) <= 1 + 1e-12 || throw(ArgumentError("rho must be in [-1,1]"))
    return abs(rho) * sqrt(cf_y * cf_d / (1 - cf_d))
end

"""
Compute θ bounds, SEs, and CIs for one coefficient across reps, then aggregate.
"""
function _sensitivity_bounds(all_theta::AbstractVector, # length n_rep
                             all_max_bias::AbstractVector,
                             all_psi_scaled::AbstractMatrix, # n × n_rep
                             all_psi_max_bias::AbstractMatrix,
                             strength::Real, level::Real)
    n_rep = length(all_theta)
    n = size(all_psi_scaled, 1)
    quant = quantile(Normal(), level)  # one-sided for bound CIs (Python uses norm.ppf(level))

    all_θ_lo = similar(all_theta)
    all_θ_hi = similar(all_theta)
    all_σ_lo = similar(all_theta)
    all_σ_hi = similar(all_theta)
    all_ci_lo = similar(all_theta)
    all_ci_hi = similar(all_theta)

    for r in 1:n_rep
        b = all_max_bias[r]
        all_θ_lo[r] = all_theta[r] - strength * b
        all_θ_hi[r] = all_theta[r] + strength * b

        ψ_lo = @view(all_psi_scaled[:, r]) .- strength .* @view(all_psi_max_bias[:, r])
        ψ_hi = @view(all_psi_scaled[:, r]) .+ strength .* @view(all_psi_max_bias[:, r])
        # psi_deriv = 1 → se = sqrt(mean(ψ²)/n)
        all_σ_lo[r] = sqrt(mean(ψ_lo .^ 2) / n)
        all_σ_hi[r] = sqrt(mean(ψ_hi .^ 2) / n)

        all_ci_lo[r] = all_θ_lo[r] - quant * all_σ_lo[r]
        all_ci_hi[r] = all_θ_hi[r] + quant * all_σ_hi[r]
    end

    # aggregate like Python (median of CIs; mean-style for theta/se via aggregate_reps pattern)
    θ_mat = reshape(all_θ_lo, 1, n_rep)
    se_mat = reshape(all_σ_lo, 1, n_rep)
    θ_lo, σ_lo = aggregate_reps(θ_mat, se_mat)
    θ_mat = reshape(all_θ_hi, 1, n_rep)
    se_mat = reshape(all_σ_hi, 1, n_rep)
    θ_hi, σ_hi = aggregate_reps(θ_mat, se_mat)

    return (
        theta_lower = θ_lo[1],
        theta_upper = θ_hi[1],
        se_lower = σ_lo[1],
        se_upper = σ_hi[1],
        ci_lower = median(all_ci_lo),
        ci_upper = median(all_ci_hi),
        all_θ_lo = all_θ_lo,
        all_θ_hi = all_θ_hi,
        all_ci_lo = all_ci_lo,
        all_ci_hi = all_ci_hi,
    )
end

"""
Robustness value: smallest `c` such that bounds with `cf_y = cf_d = c` cover `null`.
Uses golden-section search on [0, 0.999].
"""
function _robustness_value(calc_bound::Function, null::Real, theta_hat::Real;
                           which::Symbol=:theta)
    # which: :theta → use theta lower/upper; :ci → use ci lower/upper
    bound_side = null > theta_hat ? :upper : :lower
    key = if which === :ci
        bound_side === :upper ? :ci_upper : :ci_lower
    else
        bound_side === :upper ? :theta_upper : :theta_lower
    end

    # Objective: squared distance of the relevant bound to null.
    # (Do not assign to outer golden-section locals from this closure.)
    function obj(c)
        bounds = calc_bound(c)
        return (bounds[key] - null)^2
    end

    # golden-section minimize on [0, 0.999]
    lo, hi = 0.0, 0.999
    φ = (sqrt(5) - 1) / 2
    c = hi - φ * (hi - lo)
    d = lo + φ * (hi - lo)
    fc, fd = obj(c), obj(d)
    for _ in 1:60
        if fc < fd
            hi, d, fd = d, c, fc
            c = hi - φ * (hi - lo)
            fc = obj(c)
        else
            lo, c, fc = c, d, fd
            d = lo + φ * (hi - lo)
            fd = obj(d)
        end
    end
    return (lo + hi) / 2
end

"""Normalize to `Vector{SensitivityElements}`."""
function _sens_list(se)
    se === nothing && return nothing
    se isa Vector{SensitivityElements} && return se
    se isa SensitivityElements && return SensitivityElements[se]
    error("Unexpected sensitivity elements type $(typeof(se))")
end

"""
    sensitivity_analysis!(m; cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0)

Omitted-variable sensitivity analysis (Chernozhukov et al., 2022).

Requires a fitted model that stores sensitivity elements (PLR, IRM).
Supports **multiple treatments** (`null_hypothesis` may be a scalar or length-`n_coef` vector).
Results are stored in `m.sensitivity` and returned.
"""
function sensitivity_analysis!(m::AbstractDoubleML;
                               cf_y::Real=0.03,
                               cf_d::Real=0.03,
                               rho::Real=1.0,
                               level::Real=0.95,
                               null_hypothesis::Union{Real,AbstractVector{<:Real}}=0.0)
    m.fitted || error("Call fit! before sensitivity_analysis!")
    hasproperty(m, :sens_elements) || error("Sensitivity not implemented for $(typeof(m))")
    m.sens_elements === nothing && error("No sensitivity elements — fit! again with an updated package")

    els = _sens_list(m.sens_elements)
    n_coef = length(m.coef)
    length(els) == n_coef ||
        error("sens_elements length $(length(els)) ≠ n_coef=$n_coef")
    nulls = null_hypothesis isa AbstractVector ?
        Float64.(collect(null_hypothesis)) :
        fill(Float64(null_hypothesis), n_coef)
    length(nulls) == n_coef || throw(ArgumentError("null_hypothesis length must equal n_coef=$n_coef"))
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
        all_theta = vec(m.all_coef[j, :])
        for r in 1:n_rep
            mb, pmb = _max_bias_and_if(selem.sigma2[r], selem.nu2[r],
                                       @view(selem.psi_sigma2[:, r]), @view(selem.psi_nu2[:, r]))
            all_max_bias[r] = mb
            all_psi_max[:, r] = pmb
            J = mean(@view m.psi_deriv[:, r, j])
            all_psi_scaled[:, r] = @view(m.psi[:, r, j]) ./ J
        end
        bounds = _sensitivity_bounds(all_theta, all_max_bias, all_psi_scaled, all_psi_max, strength, level)
        function calc_c(c)
            s = confounding_strength(c, c, rho)
            _sensitivity_bounds(all_theta, all_max_bias, all_psi_scaled, all_psi_max, s, level)
        end
        θ̂ = m.coef[j]
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
    m.sensitivity = result
    return result
end

"""
    sensitivity_summary(m) -> String

Printable summary of the last [`sensitivity_analysis!`](@ref) run.
"""
function sensitivity_summary(m::AbstractDoubleML)
    hasproperty(m, :sensitivity) || error("Sensitivity not implemented for $(typeof(m))")
    r = m.sensitivity
    r === nothing && return "Apply sensitivity_analysis! first."
    io = IOBuffer()
    println(io, "================== Sensitivity Analysis ==================")
    println(io)
    println(io, "------------------ Scenario          ------------------")
    println(io, "Significance Level: level=$(r.level)")
    println(io, "Sensitivity parameters: cf_y=$(r.cf_y); cf_d=$(r.cf_d), rho=$(r.rho)")
    println(io)
    println(io, "------------------ Bounds with CI    ------------------")
    df = DataFrame(
        treatment = m.treat_names,
        ci_lower = r.ci_lower,
        theta_lower = r.theta_lower,
        theta = m.coef,
        theta_upper = r.theta_upper,
        ci_upper = r.ci_upper,
    )
    println(io, df)
    println(io)
    println(io, "------------------ Robustness Values ------------------")
    println(io, DataFrame(treatment=m.treat_names, rv=r.rv, rva=r.rva,
                          null_hypothesis=fill(r.null_hypothesis, length(r.rv))))
    return String(take!(io))
end

"""
    sensitivity_contour(m; cf_y_max=0.15, cf_d_max=0.15, grid_size=20,
                        rho=1.0, level=0.95, null_hypothesis=0.0, value=:theta)

Numerical sensitivity contour grid (Python `sensitivity_plot` data layer, no plotting).

# Returns
`DataFrame` with columns `cf_y`, `cf_d`, `theta_lower`, `theta_upper`,
`ci_lower`, `ci_upper`, and optionally `covers_null` for the chosen `value`.

# Arguments
- `value`: `:theta` uses theta bounds; `:ci` uses CI bounds for `covers_null`
"""
function sensitivity_contour(m::AbstractDoubleML;
                             cf_y_max::Real=0.15,
                             cf_d_max::Real=0.15,
                             grid_size::Int=20,
                             rho::Real=1.0,
                             level::Real=0.95,
                             null_hypothesis::Real=0.0,
                             value::Symbol=:theta,
                             idx_treatment::Int=1)
    m.fitted || error("Call fit! first")
    hasproperty(m, :sens_elements) || error("Sensitivity not implemented for $(typeof(m))")
    m.sens_elements === nothing && error("No sensitivity elements — re-fit")
    value in (:theta, :ci) || throw(ArgumentError("value must be :theta or :ci"))
    grid_size >= 2 || throw(ArgumentError("grid_size ≥ 2"))
    els = _sens_list(m.sens_elements)
    1 <= idx_treatment <= length(els) || throw(ArgumentError("idx_treatment out of range"))
    j = idx_treatment
    selem = els[j]
    n_rep = length(selem.sigma2)
    all_psi_scaled = similar(selem.psi_sigma2)
    all_psi_max = similar(selem.psi_sigma2)
    all_max_bias = zeros(n_rep)
    all_theta = vec(m.all_coef[j, :])
    for r in 1:n_rep
        mb, pmb = _max_bias_and_if(selem.sigma2[r], selem.nu2[r],
                                   @view(selem.psi_sigma2[:, r]), @view(selem.psi_nu2[:, r]))
        all_max_bias[r] = mb
        all_psi_max[:, r] = pmb
        J = mean(@view m.psi_deriv[:, r, j])
        all_psi_scaled[:, r] = @view(m.psi[:, r, j]) ./ J
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
        treatment = fill(m.treat_names[j], length(rows_cfy)),
    )
end

"""
    sensitivity_benchmark(dml_long, dml_short) -> NamedTuple

Benchmark confounding strength by comparing a long regression (all covariates)
to a short one (omit benchmark covariates), as in Python `sensitivity_benchmark`.

Both models must be fitted and support sensitivity elements (PLR or IRM).
Returns estimated `(cf_y, cf_d, rho, delta_theta)`.
"""
function sensitivity_benchmark(dml_long::AbstractDoubleML, dml_short::AbstractDoubleML)
    dml_long.fitted && dml_short.fitted || error("Both models must be fitted")
    dml_long.sens_elements !== nothing && dml_short.sens_elements !== nothing ||
        error("Both models need sensitivity elements")

    # Use first treatment, first-rep averages
    el_L = _sens_list(dml_long.sens_elements)[1]
    el_S = _sens_list(dml_short.sens_elements)[1]
    σ2_L = mean(el_L.sigma2)
    σ2_S = mean(el_S.sigma2)
    ν2_L = mean(el_L.nu2)
    ν2_S = mean(el_S.nu2)

    # cf_y ≈ 1 - σ2_L/σ2_S  (fraction of residual variance explained by omitted vars)
    cf_y = clamp(1 - σ2_L / max(σ2_S, eps()), 0.0, 0.999)
    # cf_d from Riesz variance gains: ν2 related to 1/E[V^2]; short has larger residual variance for D
    # gain in Riesz variance: (ν2_L - ν2_S) / ν2_L  if L includes more for treatment equation
    # Python gain_statistics uses more careful formulas; use:
    cf_d = clamp(abs(ν2_L - ν2_S) / max(max(ν2_L, ν2_S), eps()), 0.0, 0.999)

    δθ = dml_long.coef[1] - dml_short.coef[1]
    # rho sign from delta_theta and bias direction
    rho = δθ == 0 ? 1.0 : sign(δθ) * 1.0

    return (cf_y=cf_y, cf_d=cf_d, rho=rho, delta_theta=δθ,
            theta_long=dml_long.coef[1], theta_short=dml_short.coef[1])
end

"""
    sensitivity_plot(m; kwargs...)

Python-compatible alias for [`sensitivity_contour`](@ref) (returns a
numerical grid `DataFrame`; plotting is left to the user).
"""
sensitivity_plot(m::AbstractDoubleML; kwargs...) = sensitivity_contour(m; kwargs...)
