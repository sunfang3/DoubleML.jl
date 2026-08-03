"""
Abstract base for DoubleML estimators (Python `doubleml.DoubleML` analogue).
"""
abstract type AbstractDoubleML end

# Common result fields expected on concrete types after fit!:
#   coef::Vector{Float64}
#   se::Vector{Float64}
#   all_coef::Matrix{Float64}   # (n_coefs × n_rep)
#   all_se::Matrix{Float64}
#   psi::Array{Float64,3}       # (n_obs × n_rep × n_coefs)
#   fitted::Bool

function t_stat(m::AbstractDoubleML)
    m.fitted || error("Call fit! first")
    return m.coef ./ m.se
end

function pval(m::AbstractDoubleML)
    t = t_stat(m)
    return 2 .* cdf.(Normal(), -abs.(t))
end

"""
    confint(m; level=0.95, joint=false)

Pointwise normal confidence intervals. Returns a DataFrame with columns
`lower` and `upper`.
"""
function confint(m::AbstractDoubleML; level::Real=0.95, joint::Bool=false)
    m.fitted || error("Call fit! first")
    joint && error("joint confint requires bootstrap; not yet implemented in v0.1")
    α = 1 - level
    z = quantile(Normal(), 1 - α / 2)
    lower = m.coef .- z .* m.se
    upper = m.coef .+ z .* m.se
    return DataFrame(
        treatment = m.treat_names,
        lower = lower,
        upper = upper,
        level = fill(Float64(level), length(m.coef)),
    )
end

"""
    summary_table(m::AbstractDoubleML; level=0.95)

Summary table of coefficient estimates (coef, se, t, p, CI).

Named `summary_table` (not `summary`) to avoid clashing with `Base.summary`
and `StatsBase` when those packages are in scope.
"""
function summary_table(m::AbstractDoubleML; level::Real=0.95)
    m.fitted || error("Call fit! first")
    ci = confint(m; level=level)
    return DataFrame(
        treatment = m.treat_names,
        coef = m.coef,
        std_err = m.se,
        t = t_stat(m),
        pvalue = pval(m),
        ci_lower = ci.lower,
        ci_upper = ci.upper,
    )
end

# Convenience alias used like Python's `.summary` when unambiguous
const dml_summary = summary_table

function Base.show(io::IO, m::AbstractDoubleML)
    name = string(typeof(m).name.name)
    if !m.fitted
        print(io, "$name (not fitted)")
    else
        print(io, "$name(coef=$(round.(m.coef; digits=4)), se=$(round.(m.se; digits=4)))")
    end
end

# ---- score helpers (linear scores ψ = ψ_a θ + ψ_b) ----

"""
Estimate θ from linear score elements: θ = -mean(ψ_b) / mean(ψ_a).
"""
function est_coef_linear(psi_a::AbstractVector, psi_b::AbstractVector)
    ma = mean(psi_a)
    abs(ma) < 1e-14 && error("Degenerate score: mean(ψ_a) ≈ 0")
    return -mean(psi_b) / ma
end

"""
Standard error for linear DML score (iid case):

    se = sqrt( mean(ψ²) / mean(ψ_a)² / n )

where ψ_i = ψ_a,i * θ + ψ_b,i.
"""
function se_linear(psi_a::AbstractVector, psi_b::AbstractVector, θ::Real)
    n = length(psi_a)
    ψ = psi_a .* θ .+ psi_b
    J = mean(psi_a)           # ∂/∂θ E[ψ] = E[ψ_a]
    σ2 = mean(ψ .^ 2)
    return sqrt(σ2 / (J^2) / n)
end

"""
Aggregate coefficients across repeated sample splits (median of coefs,
mean of squared SEs for variance).
Python DoubleML uses mean of coefs and a specific SE aggregation;
we use the same mean-aggregation as the Python package's default.
"""
function aggregate_reps(all_coef::AbstractMatrix, all_se::AbstractMatrix)
    # all_* : n_coefs × n_rep
    coef = vec(mean(all_coef; dims=2))
    # SE aggregation: sqrt(mean(se²) + sample var of coefs)  — conservative
    # Python uses a refined formula; for n_rep=1 this equals se.
    n_rep = size(all_coef, 2)
    if n_rep == 1
        se = vec(all_se)
    else
        se2 = vec(mean(all_se .^ 2; dims=2))
        # add between-split variability
        if n_rep > 1
            between = vec(var(all_coef; dims=2, corrected=true))
            se = sqrt.(se2 .+ between)
        else
            se = sqrt.(se2)
        end
    end
    return coef, se
end
