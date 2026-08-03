# Best Linear Predictor (BLP) for CATE / GATE
# Mirrors Python doubleml.utils.blp.DoubleMLBLP (Chernozhukov et al. heterogeneous TE)

"""
    DoubleMLBLP

Best linear predictor of an orthogonal signal on a user-supplied basis.

Used after `fit!` via [`cate`](@ref) / [`gate`](@ref) on PLR or IRM models:

```
θ̂(x) ≈ β̂' φ(x)
```

where `φ` is the basis (e.g. group dummies for GATE, polynomials/splines for CATE).
"""
mutable struct DoubleMLBLP
    orth_signal::Matrix{Float64}           # n × n_rep
    basis_list::Vector{Matrix{Float64}}    # length n_rep; each n × d
    basis_names::Vector{String}
    is_gate::Bool
    all_coef::Matrix{Float64}              # d × n_rep
    all_se::Matrix{Float64}
    omega::Array{Float64,3}                # d × d × n_rep  (HC0 cov)
    coef::Vector{Float64}
    se::Vector{Float64}
    fitted::Bool
end

function Base.show(io::IO, m::DoubleMLBLP)
    if !m.fitted
        print(io, "DoubleMLBLP(not fitted; d=$(length(m.basis_names)), is_gate=$(m.is_gate))")
    else
        print(io, "DoubleMLBLP(coef=$(round.(m.coef; digits=4)), se=$(round.(m.se; digits=4)))")
    end
end

# ---- basis / groups helpers -------------------------------------------------

"""
    _as_basis_matrix(basis) -> (Matrix{Float64}, Vector{String})

Accept a matrix or DataFrame as BLP basis.
"""
function _as_basis_matrix(basis::AbstractMatrix)
    X = Float64.(basis)
    names = ["b$i" for i in 1:size(X, 2)]
    return X, names
end

function _as_basis_matrix(basis::DataFrame)
    X = Matrix{Float64}(basis)
    names = string.(names(basis))
    return X, names
end

"""
    group_dummies(groups) -> (Matrix{Float64}, Vector{String})

Convert group labels or an existing dummy matrix to a one-hot design.

# Arguments
- `groups`: length-`n` vector of labels (`String`, `Int`, …), an `n×K` dummy
  matrix (0/1 or Bool), or a one-column `DataFrame` of labels / multi-column dummies.
"""
function group_dummies(groups::AbstractVector)
    n = length(groups)
    n >= 1 || throw(ArgumentError("groups is empty"))
    # preserve first-seen order of levels
    levels = unique(groups)
    K = length(levels)
    K >= 1 || throw(ArgumentError("no group levels found"))
    X = zeros(Float64, n, K)
    names = Vector{String}(undef, K)
    for (k, lev) in enumerate(levels)
        X[:, k] = Float64.(groups .== lev)
        names[k] = "Group_$lev"
    end
    return X, names
end

function group_dummies(groups::AbstractMatrix)
    X = Float64.(groups)
    n, K = size(X)
    # treat as already dummy-coded if all entries in {0,1}
    vals = unique(vec(X))
    all(v -> v == 0.0 || v == 1.0, vals) ||
        throw(ArgumentError("group matrix must be dummy-coded (0/1)"))
    names = ["Group_$k" for k in 1:K]
    return X, names
end

function group_dummies(groups::DataFrame)
    if ncol(groups) == 1
        return group_dummies(groups[!, 1])
    end
    X, _ = _as_basis_matrix(groups)
    # if already 0/1, keep column names
    vals = unique(vec(X))
    all(v -> v == 0.0 || v == 1.0, vals) ||
        throw(ArgumentError("multi-column groups DataFrame must be dummy-coded (0/1)"))
    return X, string.(names(groups))
end

"""
    poly_basis(x; degree=3, include_intercept=true) -> Matrix{Float64}

Polynomial basis in a single continuous feature (for simple CATE demos).
Columns: `1, x, x², …, x^degree` (intercept optional).
"""
function poly_basis(x::AbstractVector; degree::Int=3, include_intercept::Bool=true)
    degree >= 1 || throw(ArgumentError("degree must be ≥ 1"))
    n = length(x)
    xf = Float64.(x)
    cols = include_intercept ? (0:degree) : (1:degree)
    Φ = ones(Float64, n, length(cols))
    for (j, p) in enumerate(cols)
        p == 0 && continue
        Φ[:, j] = xf .^ p
    end
    return Φ
end

function poly_basis(X::AbstractMatrix; degree::Int=2, include_intercept::Bool=true)
    # additive polynomials across columns (no interactions) — lightweight helper
    n, p = size(X)
    blocks = Matrix{Float64}[]
    if include_intercept
        push!(blocks, ones(n, 1))
    end
    for j in 1:p
        for d in 1:degree
            push!(blocks, Float64.(X[:, j]) .^ d)
        end
    end
    return hcat(blocks...)
end

# ---- OLS + HC0 --------------------------------------------------------------

"""
OLS with HC0 (White) sandwich covariance. No intercept is added — the basis
should already include an intercept column if desired.
"""
function _ols_hc0(y::AbstractVector, X::AbstractMatrix)
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("y and X row count differ"))
    p >= 1 || throw(ArgumentError("basis has zero columns"))

    XtX = X' * X
    # rank check
    F = svd(XtX)
    if minimum(F.S) < 1e-12 * maximum(F.S)
        error("Singular BLP design matrix (collinear basis / empty groups). " *
              "Check group sizes and basis rank.")
    end
    β = XtX \ (X' * Float64.(y))
    e = Float64.(y) .- X * β
    # meat = X' diag(e²) X
    meat = X' * (X .* (e .^ 2))
    bread = inv(XtX)
    Ω = bread * meat * bread
    # numerical symmetry
    Ω = (Ω + Ω') / 2
    se = sqrt.(max.(diag(Ω), 0.0))
    return β, se, Ω
end

function _psd_factor(Ω::AbstractMatrix)
    # A with A*A' ≈ Ω for multiplier draws
    Ωs = Symmetric((Ω + Ω') / 2)
    try
        return Matrix(cholesky(Ωs; check=true).L)
    catch
        F = eigen(Ωs)
        λ = sqrt.(clamp.(F.values, 0.0, Inf))
        return F.vectors * Diagonal(λ)
    end
end

# ---- construct / fit BLP ----------------------------------------------------

"""
    DoubleMLBLP(orth_signal, basis; is_gate=false)

Construct an unfitted BLP. `orth_signal` is `n` or `n × n_rep`;
`basis` is an `n × d` matrix / DataFrame, or a vector of length `n_rep` of those
(as used by PLR CATE, where the basis is residual-weighted per split).
"""
function DoubleMLBLP(orth_signal::AbstractVecOrMat, basis; is_gate::Bool=false)
    S = orth_signal isa AbstractVector ? reshape(Float64.(orth_signal), :, 1) :
        Float64.(orth_signal)
    n, n_rep = size(S)

    if basis isa AbstractVector && !(eltype(basis) <: Real)
        # list of bases (one per rep)
        length(basis) == n_rep ||
            throw(ArgumentError("basis list length must equal n_rep=$n_rep"))
        mats = Matrix{Float64}[]
        names_ref = nothing
        for (i, b) in enumerate(basis)
            M, nm = _as_basis_matrix(b)
            size(M, 1) == n || throw(DimensionMismatch("basis[$i] has wrong n"))
            if names_ref === nothing
                names_ref = nm
            elseif nm != names_ref && size(M, 2) != length(names_ref)
                throw(ArgumentError("per-rep bases must share the same dimension"))
            end
            push!(mats, M)
        end
        d = size(mats[1], 2)
        names = names_ref === nothing ? ["b$i" for i in 1:d] : names_ref
    else
        M, names = _as_basis_matrix(basis)
        size(M, 1) == n || throw(DimensionMismatch("basis and signal have different n"))
        mats = [M for _ in 1:n_rep]
        d = size(M, 2)
    end

    return DoubleMLBLP(
        S, mats, names, is_gate,
        fill(NaN, d, n_rep), fill(NaN, d, n_rep), fill(NaN, d, d, n_rep),
        Float64[], Float64[], false,
    )
end

"""
    fit!(blp::DoubleMLBLP) -> DoubleMLBLP

Estimate BLP coefficients by OLS with HC0 standard errors, one model per
cross-fitting repetition, then aggregate.
"""
function fit!(m::DoubleMLBLP)
    n_rep = size(m.orth_signal, 2)
    d = size(m.basis_list[1], 2)
    for r in 1:n_rep
        β, se, Ω = _ols_hc0(@view(m.orth_signal[:, r]), m.basis_list[r])
        m.all_coef[:, r] = β
        m.all_se[:, r] = se
        m.omega[:, :, r] = Ω
    end
    coef, se = aggregate_reps(m.all_coef, m.all_se)
    m.coef = coef
    m.se = se
    m.fitted = true
    return m
end

function t_stat(m::DoubleMLBLP)
    m.fitted || error("Call fit! first")
    return m.coef ./ m.se
end

function pval(m::DoubleMLBLP)
    t = t_stat(m)
    return 2 .* cdf.(Normal(), -abs.(t))
end

function summary_table(m::DoubleMLBLP; level::Real=0.95)
    m.fitted || error("Call fit! first")
    ci = confint(m; level=level, joint=false)
    return DataFrame(
        term = m.basis_names,
        coef = m.coef,
        std_err = m.se,
        t = t_stat(m),
        pvalue = pval(m),
        ci_lower = ci.lower,
        ci_upper = ci.upper,
    )
end

"""
    confint(blp; basis=nothing, joint=false, level=0.95, n_rep_boot=500, rng=...)

Confidence intervals for BLP coefficients or for predictions `basis * β`.

- `basis=nothing` and `is_gate=true` → group-level GATE CIs (identity design).
- `basis=nothing` and `is_gate=false` → CIs for the BLP coefficients β.
- `basis` given → CIs for the predicted effect surface at those rows.

`joint=true` uses a Gaussian multiplier bootstrap over the estimated HC0
covariance (simultaneous bands), matching Python DoubleMLBLP.
"""
function confint(m::DoubleMLBLP; basis=nothing, joint::Bool=false,
                 level::Real=0.95, n_rep_boot::Int=500,
                 rng::AbstractRNG=Random.default_rng())
    m.fitted || error("Call fit! before confint")
    (0 < level < 1) || throw(ArgumentError("level must be in (0,1)"))
    α = 1 - level
    n_rep = size(m.all_coef, 2)
    d = length(m.coef)

    # --- default: coefficients or GATE table ---
    if basis === nothing
        if m.is_gate
            # identity → group effects = coefficients
            B = Matrix{Float64}(I, d, d)
            row_names = m.basis_names
        else
            # pointwise CI on β (median over reps)
            z = quantile(Normal(), 1 - α / 2)
            # per-rep CI then median aggregate
            lo_reps = m.all_coef .- z .* m.all_se
            hi_reps = m.all_coef .+ z .* m.all_se
            return DataFrame(
                term = m.basis_names,
                lower = vec(median(lo_reps; dims=2)),
                effect = m.coef,
                upper = vec(median(hi_reps; dims=2)),
                level = fill(Float64(level), d),
                joint = fill(false, d),
            )
        end
    else
        B, row_names_raw = _as_basis_matrix(basis)
        size(B, 2) == d || throw(DimensionMismatch("basis must have $d columns"))
        row_names = row_names_raw  # overwritten for row index below
        row_names = ["i=$i" for i in 1:size(B, 1)]
    end

    n_b = size(B, 1)
    all_g = fill(NaN, n_b, n_rep)
    all_se = fill(NaN, n_b, n_rep)
    for r in 1:n_rep
        β = @view m.all_coef[:, r]
        Ω = @view m.omega[:, :, r]
        all_g[:, r] = B * β
        # se_i = sqrt(b_i' Ω b_i)
        BΩ = B * Ω
        all_se[:, r] = sqrt.(max.(sum(BΩ .* B; dims=2), 0.0))
    end
    g_hat, _ = aggregate_reps(all_g, all_se)

    if joint
        crit = zeros(n_rep)
        for r in 1:n_rep
            A = _psd_factor(@view m.omega[:, :, r])  # d × d
            # samples: B * A * Z  with Z ~ N(0,I_d), size n_b × n_rep_boot
            Z = randn(rng, d, n_rep_boot)
            draws = B * (A * Z)                      # n_b × n_rep_boot
            se_r = @view all_se[:, r]
            tboot = abs.(draws ./ se_r)              # broadcast rows
            max_t = vec(maximum(tboot; dims=1))
            crit[r] = quantile(max_t, level)
        end
    else
        z = quantile(Normal(), 1 - α / 2)
        crit = fill(z, n_rep)
    end

    lo_reps = all_g .- reshape(crit, 1, n_rep) .* all_se
    hi_reps = all_g .+ reshape(crit, 1, n_rep) .* all_se
    lower = vec(median(lo_reps; dims=2))
    upper = vec(median(hi_reps; dims=2))

    return DataFrame(
        term = basis === nothing && m.is_gate ? m.basis_names : row_names,
        lower = lower,
        effect = g_hat,
        upper = upper,
        level = fill(Float64(level), n_b),
        joint = fill(joint, n_b),
    )
end

# ---- extract orthogonal signals from fitted models --------------------------

"""IRM ATE orthogonal signal: ψ_b = DR score (n × n_rep)."""
function _orth_signal_irm(m::DoubleMLIRM)
    m.fitted || error("Call fit! before cate/gate")
    m.score == "ATE" ||
        throw(ArgumentError("IRM cate/gate currently requires score=\"ATE\" (got $(m.score))"))
    n, n_rep, _ = size(m.psi)
    # ψ = ψ_a θ + ψ_b with ψ_a ≡ -1  ⇒  ψ_b = ψ + θ
    S = similar(m.psi, n, n_rep)
    for r in 1:n_rep
        θ = m.all_coef[1, r]
        S[:, r] = @view(m.psi[:, r, 1]) .+ θ
    end
    return S
end

"""PLR partialled-out residuals (Ỹ, D̃), each n × n_rep."""
function _partial_out_plr(m::DoubleMLPLR)
    m.fitted || error("Call fit! before cate/gate")
    isempty(m.predictions) &&
        error("predictions missing; call fit!(m; store_predictions=true)")
    y = m.data.y
    d = m.data.d
    n_rep = m.n_rep
    n = length(y)
    Yt = fill(NaN, n, n_rep)
    Dt = fill(NaN, n, n_rep)
    m_hat = m.predictions["ml_m"]
    if m.score == "partialling out"
        l_hat = m.predictions["ml_l"]
        for r in 1:n_rep
            Yt[:, r] = y .- @view(l_hat[:, r])
            Dt[:, r] = d .- @view(m_hat[:, r])
        end
    else
        # IV-type: Ỹ ≈ y − ĝ − θ m̂  (= y − E[Y|X] under constant θ)
        haskey(m.predictions, "ml_g") ||
            error("IV-type PLR needs ml_g predictions; re-fit with store_predictions=true")
        g_hat = m.predictions["ml_g"]
        for r in 1:n_rep
            θ = m.all_coef[1, r]
            Yt[:, r] = y .- θ .* @view(m_hat[:, r]) .- @view(g_hat[:, r])
            Dt[:, r] = d .- @view(m_hat[:, r])
        end
    end
    return Yt, Dt
end

# ---- public cate / gate API -------------------------------------------------

"""
    cate(m, basis; is_gate=false) -> DoubleMLBLP

Conditional average treatment effects via best linear projection of the
orthogonal signal onto `basis` (n × d matrix or DataFrame).

- **IRM (ATE score):** regress the doubly-robust scores on `basis`.
- **PLR:** residual-on-residual BLP — regress `Ỹ` on `basis ⊙ D̃`
  (each basis column multiplied by the treatment residual), so that
  `θ(x) ≈ β' φ(x)`.

Returns a fitted [`DoubleMLBLP`](@ref). Use `summary_table` / `confint`.
"""
function cate(m::DoubleMLIRM, basis; is_gate::Bool=false)
    S = _orth_signal_irm(m)
    blp = DoubleMLBLP(S, basis; is_gate=is_gate)
    return fit!(blp)
end

function cate(m::DoubleMLPLR, basis; is_gate::Bool=false)
    Yt, Dt = _partial_out_plr(m)
    n_rep = size(Yt, 2)
    # shared or per-rep raw basis
    if basis isa AbstractVector && !(eltype(basis) <: Real)
        length(basis) == n_rep ||
            throw(ArgumentError("basis list length must equal n_rep=$n_rep"))
        raw_list = basis
        names_ref = nothing
        weighted = Matrix{Float64}[]
        for r in 1:n_rep
            M, nm = _as_basis_matrix(raw_list[r])
            names_ref = names_ref === nothing ? nm : names_ref
            push!(weighted, M .* Dt[:, r])          # broadcast: each col × D̃
        end
        blp = DoubleMLBLP(Yt, weighted; is_gate=is_gate)
        blp.basis_names = names_ref
    else
        M, nm = _as_basis_matrix(basis)
        weighted = [M .* Dt[:, r] for r in 1:n_rep]
        blp = DoubleMLBLP(Yt, weighted; is_gate=is_gate)
        blp.basis_names = nm
    end
    return fit!(blp)
end

"""
    gate(m, groups) -> DoubleMLBLP

Group average treatment effects. `groups` may be:

- a length-`n` vector of labels,
- an `n × K` dummy matrix,
- a one-column or dummy-coded `DataFrame`.

Groups should be mutually exclusive for a clean interpretation.
Returns a fitted [`DoubleMLBLP`](@ref) with `is_gate=true`.
"""
function gate(m::Union{DoubleMLIRM,DoubleMLPLR}, groups)
    G, gnames = group_dummies(groups)
    # warn on tiny groups
    ns = vec(sum(G; dims=1))
    if any(ns .<= 5)
        @warn "At least one group has ≤ 5 observations; GATE estimates may be unstable" group_sizes=ns
    end
    blp = cate(m, G; is_gate=true)
    blp.basis_names = gnames
    return blp
end
