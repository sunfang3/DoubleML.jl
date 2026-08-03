# Regression Discontinuity — RDFlex-style (Python doubleml.rdd.RDFlex)
# Residualization of Y (and D if fuzzy) with ML, then local linear RD.
# No rdrobust dependency: bandwidth via ROT / iterative residual ROT.

"""
    DoubleMLRDD

Sharp or fuzzy RDD with ML residualization near the cutoff, then weighted
local linear estimation. Aligns with Python `RDFlex` (without `rdrobust`).

# Keywords
- `cutoff` — cutoff on the running variable
- `fuzzy` — fuzzy design (needs `ml_m`)
- `h` / `h_fs` — fixed final bandwidth / initial first-stage bandwidth (`NaN` → ROT)
- `fs_kernel` — `"triangular"` | `"uniform"` | `"epanechnikov"`
- `fs_specification` — `"cutoff"` | `"cutoff and score"` | `"interacted cutoff and score"`
- `n_iterations` — iterative bandwidth updates (default 2, as Python)

# Data
[`DoubleMLData`](@ref) with `score` (running variable), `y`, `d`, `x`.
"""
mutable struct DoubleMLRDD <: AbstractDoubleML
    data::DoubleMLData
    ml_g::Any
    ml_m::Any
    cutoff::Float64
    fuzzy::Bool
    n_folds::Int
    n_rep::Int
    h::Float64                 # fixed final bandwidth if provided
    h_fs::Float64              # initial first-stage bandwidth
    fs_kernel::String
    fs_specification::String
    n_iterations::Int
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
    h_used::Float64
    all_h::Vector{Float64}
end

const _FS_SPECS = ("cutoff", "cutoff and score", "interacted cutoff and score")

function DoubleMLRDD(data::DoubleMLData, ml_g, ml_m=nothing;
                     cutoff::Real=0.0,
                     fuzzy::Bool=false,
                     n_folds::Int=5,
                     n_rep::Int=1,
                     h::Real=NaN,
                     h_fs::Real=NaN,
                     fs_kernel::AbstractString="triangular",
                     fs_specification::AbstractString="cutoff",
                     n_iterations::Int=2,
                     draw_sample_splitting::Bool=true,
                     rng::AbstractRNG=Random.default_rng())
    data.score === nothing && throw(ArgumentError("RDD requires running variable in data.score"))
    fuzzy && ml_m === nothing && throw(ArgumentError("fuzzy RDD requires ml_m"))
    fs_kernel in ("triangular", "uniform", "epanechnikov") ||
        throw(ArgumentError("fs_kernel must be triangular/uniform/epanechnikov"))
    fs_specification in _FS_SPECS ||
        throw(ArgumentError("fs_specification must be one of $_FS_SPECS"))
    n_iterations >= 1 || throw(ArgumentError("n_iterations ≥ 1"))

    smpls = draw_sample_splitting ?
        make_repeated_folds(n_obs(data), n_folds, n_rep; rng=rng) :
        Vector{Any}()
    n = n_obs(data)
    return DoubleMLRDD(
        data, ml_g, ml_m, Float64(cutoff), fuzzy, n_folds, n_rep,
        Float64(h), Float64(h_fs), String(fs_kernel), String(fs_specification),
        n_iterations, smpls,
        Float64[], Float64[],
        zeros(1, n_rep), zeros(1, n_rep),
        fill(NaN, n, n_rep, 1), fill(NaN, n, n_rep, 1),
        Dict{String,Any}(), ["LATE_RD"],
        nothing, false, rng, NaN, fill(NaN, n_rep),
    )
end

"""Alias matching Python class name."""
const RDFlex = DoubleMLRDD

function _kernel_w(u::AbstractVector, kernel::String)
    a = abs.(u)
    if kernel == "triangular"
        return ifelse.(a .< 1, 1 .- a, 0.0)
    elseif kernel == "uniform"
        return ifelse.(a .< 1, 1.0, 0.0)
    else  # epanechnikov
        return ifelse.(a .< 1, 0.75 .* (1 .- a .^ 2), 0.0)
    end
end

function _rot_bandwidth(score::AbstractVector, cutoff::Real)
    s = score .- cutoff
    n = length(s)
    σ = std(s)
    return 1.84 * σ * n^(-1 / 5)
end

"""Build first-stage design features Z and left/right evaluation points (at cutoff)."""
function _fs_design(s::AbstractVector, spec::String)
    z = Float64.(s .>= 0)  # intended treatment
    n = length(s)
    if spec == "cutoff"
        Z = reshape(z, n, 1)
        ZL = zeros(n, 1)
        ZR = ones(n, 1)
    elseif spec == "cutoff and score"
        Z = hcat(z, s)
        ZL = hcat(zeros(n), s)           # left: T=0, keep score
        ZR = hcat(ones(n), zeros(n))     # at cutoff from right: T=1, score=0
        # Python: Z_left = zeros_like(Z); Z_right = [1, 0] for score at 0
        ZL = zeros(n, 2)
        ZR = hcat(ones(n), zeros(n))
    else  # interacted cutoff and score
        Z = hcat(z, z .* s, s)
        ZL = zeros(n, 3)
        ZR = hcat(ones(n), zeros(n), zeros(n))
    end
    return Z, ZL, ZR, z
end

"""Cross-fit η = (μ_left + μ_right)/2 with sample weights and fs design."""
function _fit_nuisance_rdd(ml, outcome, X, Z, ZL, ZR, weights, folds; classifier::Bool=false)
    n = length(outcome)
    muL = fill(NaN, n)
    muR = fill(NaN, n)
    ZX = hcat(Z, X)
    ZXL = hcat(ZL, X)
    ZXR = hcat(ZR, X)
    for (train, test) in folds
        # drop zero-weight train points for stability
        tr = train[weights[train] .> 0]
        length(tr) < 5 && continue
        m = clone(ml)
        # weighted fit via case weights replication is heavy; use weighted least squares
        # for generic learners: fit on positive-weight rows (approx)
        fit!(m, ZX[tr, :], outcome[tr])
        if classifier || is_classifier(m)
            muL[test] = predict_proba(m, ZXL[test, :])
            muR[test] = predict_proba(m, ZXR[test, :])
        else
            muL[test] = predict(m, ZXL[test, :])
            muR[test] = predict(m, ZXR[test, :])
        end
    end
    for i in 1:n
        if !isfinite(muL[i]); muL[i] = isfinite(muR[i]) ? muR[i] : 0.0; end
        if !isfinite(muR[i]); muR[i] = isfinite(muL[i]) ? muL[i] : 0.0; end
    end
    return 0.5 .* (muL .+ muR)
end

"""Local linear weighted regression of y on [1, s, d, d*s]."""
function _local_linear_rd(s, y, d, w)
    n = length(y)
    X = hcat(ones(n), s, d, d .* s)
    sw = sqrt.(max.(w, 0.0))
    Xw = X .* sw
    yw = y .* sw
    keep = sw .> 0
    sum(keep) < 8 && error("Too few observations inside bandwidth for RDD")
    Xk = Xw[keep, :]
    yk = yw[keep]
    β = Xk \ yk
    resid = y .- X * β
    bread = inv(Xk' * Xk)
    meat = Xk' * (Xk .* (resid[keep] .^ 2))
    Ω = bread * meat * bread
    se = sqrt(max(Ω[3, 3], 0.0))
    return β[3], se, resid
end

"""Simple IK-style / ROT bandwidth on residualized outcome near cutoff."""
function _bandwidth_from_residuals(s::AbstractVector, My::AbstractVector; h_hint::Real=NaN)
    # use ROT on running variable; optionally shrink toward residual scale
    h0 = isnan(h_hint) ? _rot_bandwidth(s, 0.0) : h_hint
    # local residual variance near cutoff
    near = abs.(s) .< max(h0, eps())
    if count(near) >= 20
        σ = std(@view My[near])
        n = count(near)
        # Silverman on residualized outcome density scale
        h1 = 1.84 * max(σ, 1e-6) * n^(-1 / 5)
        # geometric blend
        return sqrt(h0 * max(h1, 0.25 * h0))
    end
    return h0
end

function fit!(m::DoubleMLRDD; store_predictions::Bool=true)
    data = m.data
    X, y = data.x, data.y
    s_raw = data.score
    cutoff = m.cutoff
    s = s_raw .- cutoff  # center at 0
    n = n_obs(data)
    n_rep = m.n_rep

    Z, ZL, ZR, z_int = _fs_design(s, m.fs_specification)
    d = m.fuzzy ? data.d : z_int

    h_fs = isnan(m.h_fs) ? _rot_bandwidth(s_raw, cutoff) : m.h_fs
    fixed_h = !isnan(m.h)

    if isempty(m.smpls)
        m.smpls = make_repeated_folds(n, m.n_folds, n_rep; rng=m.rng)
    end

    all_coef = zeros(1, n_rep)
    all_se = zeros(1, n_rep)
    all_h = fill(NaN, n_rep)
    psi_arr = fill(NaN, n, n_rep, 1)
    psi_d_arr = fill(NaN, n, n_rep, 1)
    My_store = fill(NaN, n, n_rep)

    for r in 1:n_rep
        folds = m.smpls[r]
        h_cur = h_fs
        weights = _kernel_w(s ./ h_cur, m.fs_kernel)
        My = similar(y)
        Md = similar(d)

        for it in 1:m.n_iterations
            weights = _kernel_w(s ./ h_cur, m.fs_kernel)
            ηY = _fit_nuisance_rdd(m.ml_g, y, X, Z, ZL, ZR, weights, folds; classifier=false)
            My = y .- ηY
            if m.fuzzy
                ηD = _fit_nuisance_rdd(m.ml_m, d, X, Z, ZL, ZR, weights, folds;
                                       classifier=is_classifier(m.ml_m))
                Md = d .- ηD
            end

            if it < m.n_iterations && !fixed_h
                h_cur = _bandwidth_from_residuals(s, My; h_hint=h_cur)
            end
        end

        # final local linear on residualized outcomes with final bandwidth
        h_final = fixed_h ? m.h : h_cur
        w_final = _kernel_w(s ./ h_final, m.fs_kernel)
        if m.fuzzy
            rf, se_rf, _ = _local_linear_rd(s, My, z_int, w_final)
            fs, se_fs, _ = _local_linear_rd(s, Md, z_int, w_final)
            abs(fs) < 1e-8 && error("Weak first stage in fuzzy RDD")
            θ = rf / fs
            se = abs(θ) * sqrt((se_rf / max(abs(rf), 1e-12))^2 + (se_fs / fs)^2)
            resid = My .- θ .* Md
        else
            θ, se, resid = _local_linear_rd(s, My, z_int, w_final)
        end

        all_coef[1, r] = θ
        all_se[1, r] = se
        all_h[r] = h_final
        My_store[:, r] = My
        ψ = w_final .* resid
        psi_arr[:, r, 1] = ψ .- mean(ψ)
        psi_d_arr[:, r, 1] .= -1.0
    end

    coef, se = aggregate_reps(all_coef, all_se)
    m.coef = coef; m.se = se
    m.all_coef = all_coef; m.all_se = all_se
    m.psi = psi_arr; m.psi_deriv = psi_d_arr
    m.all_h = all_h
    m.h_used = median(all_h)
    m.boot = nothing
    if store_predictions
        m.predictions = Dict("M_Y" => My_store)
    end
    m.fitted = true
    return m
end
