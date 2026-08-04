# Multiplier bootstrap (Python DoubleMLFramework.bootstrap analogue)
# BootstrapResult is defined in base.jl (forward declaration for field types).

function _draw_weights(method::AbstractString, n_rep_boot::Int, n_obs::Int, rng::AbstractRNG)
    method in ("normal", "Bayes", "wild") ||
        throw(ArgumentError("bootstrap method must be \"normal\", \"Bayes\", or \"wild\""))
    if method == "normal"
        return randn(rng, n_rep_boot, n_obs)
    elseif method == "Bayes"
        return rand(rng, Exponential(1.0), n_rep_boot, n_obs) .- 1
    else
        xx = randn(rng, n_rep_boot, n_obs)
        yy = randn(rng, n_rep_boot, n_obs)
        return xx ./ sqrt(2) .+ (yy .^ 2 .- 1) ./ 2
    end
end

"""
    bootstrap!(m; method="normal", n_rep_boot=500)

Multiplier bootstrap for joint confidence intervals, matching Python DoubleML.

Requires `fit!` first. Uses stored `psi` and `psi_deriv` (score and ∂ψ/∂θ).

After calling, use `confint(m; joint=true)`.
"""
function bootstrap!(m::AbstractDoubleML; method::AbstractString="normal",
                    n_rep_boot::Int=500, rng::Union{Nothing,AbstractRNG}=nothing)
    m.fitted || error("Call fit! before bootstrap!")
    n_rep_boot >= 1 || throw(ArgumentError("n_rep_boot must be ≥ 1"))
    hasproperty(m, :psi_deriv) || error("Model does not store psi_deriv; re-fit with updated package")
    m.psi_deriv === nothing && error("psi_deriv is missing; call fit! again")

    rng = rng === nothing ? m.rng : rng
    n = size(m.psi, 1)
    n_rep = size(m.psi, 2)
    n_coef = size(m.psi, 3)

    boot_t = fill(NaN, n_rep_boot, n_coef, n_rep)

    for r in 1:n_rep
        # The same multiplier draw must be used for every coefficient in a
        # replication.  Re-drawing inside the coefficient loop destroys the
        # cross-coefficient dependence needed for joint CIs and Romano–Wolf.
        weights = _draw_weights(method, n_rep_boot, n, rng)
        # scaled_psi = psi / mean(psi_deriv); var_scaling = n * se_rep
        for j in 1:n_coef
            ψ = @view m.psi[:, r, j]
            ψd = @view m.psi_deriv[:, r, j]
            J = mean(ψd)
            abs(J) < 1e-14 && error("Degenerate psi_deriv in bootstrap")
            scaled = ψ ./ J
            se_r = m.all_se[j, r]
            # Python: weights @ (scaled_psi / (n * se))
            # But var_scaling_factors is n, so / (n * se)
            # Wait: matmul(weights, scaled/var_scaling) where weights ~ N(0,1)
            # Actually looking again: var_scaling = n * se, and result is boot t-stat
            # E[sum w_i * psi_scaled_i / (n se)] has variance mean(psi_scaled^2)/(n se^2)
            # and se = sqrt(mean(psi^2)/(J^2 n)) so mean(psi_scaled^2)/n = se^2
            # thus var of boot t is se^2 / se^2 = 1. Good.
            denom = n * se_r
            boot_t[:, j, r] = weights * (scaled ./ denom)
        end
    end

    m.boot = BootstrapResult(String(method), n_rep_boot, boot_t)
    return m
end
