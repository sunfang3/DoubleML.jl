"""
    make_plr_data(; n_obs=500, dim_x=20, theta=0.5, seed=nothing) -> DoubleMLData

Synthetic PLR data in the spirit of Chernozhukov et al. / Python
`make_plr_CCDDHNR2018`.
"""
function make_plr_data(; n_obs::Int=500, dim_x::Int=20, theta::Real=0.5,
                       return_type::Symbol=:DoubleMLData,
                       seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    X = randn(rng, n_obs, dim_x)
    # nonlinear baseline
    b = X[:, 1] .+ 0.25 .* (X[:, 3] .^ 2)  # simple confounder function
    # propensity / treatment process
    m = 0.5 .* X[:, 1] .+ 0.5 .* X[:, 2]
    d = m .+ randn(rng, n_obs)
    y = theta .* d .+ b .+ randn(rng, n_obs)

    data = DoubleMLData(X, y, d; y_col="y", d_col="d")
    if return_type === :DataFrame
        df = DataFrame(X, :auto)
        rename!(df, ["X$i" for i in 1:dim_x])
        df.y = y
        df.d = d
        return df
    end
    return data
end

"""
    make_plr_multi_data(; n_obs=800, dim_x=8, theta=[0.5, -0.3], seed=nothing)

PLR DGP with multiple continuous treatments. True slopes ≈ `theta`.
"""
function make_plr_multi_data(; n_obs::Int=800, dim_x::Int=8,
                             theta::AbstractVector=[0.5, -0.3],
                             seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    n_t = length(theta)
    n_t >= 1 || throw(ArgumentError("theta non-empty"))
    dim_x >= 2 || throw(ArgumentError("dim_x ≥ 2"))
    X = randn(rng, n_obs, dim_x)
    b = X[:, 1] .+ 0.25 .* (X[:, 2] .^ 2)
    D = zeros(n_obs, n_t)
    for j in 1:n_t
        D[:, j] = 0.4 .* X[:, min(j, dim_x)] .+ randn(rng, n_obs)
    end
    y = D * Float64.(theta) .+ b .+ randn(rng, n_obs)
    return DoubleMLData(X, y, D; y_col="y", d_cols=["d$j" for j in 1:n_t])
end

"""
    make_plr_cluster_data(; n_obs=600, n_clusters=40, dim_x=5, theta=0.5, seed=nothing)

One-way clustered PLR DGP (shared cluster random effect).
"""
function make_plr_cluster_data(; n_obs::Int=600, n_clusters::Int=40, dim_x::Int=5,
                               theta::Real=0.5, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    n_clusters >= 4 || throw(ArgumentError("n_clusters ≥ 4"))
    cluster = repeat(1:n_clusters, inner=n_obs ÷ n_clusters)
    # pad if needed
    while length(cluster) < n_obs
        push!(cluster, n_clusters)
    end
    cluster = cluster[1:n_obs]
    X = randn(rng, n_obs, dim_x)
    α = randn(rng, n_clusters)
    a = α[cluster]
    d = 0.5 .* X[:, 1] .+ 0.3 .* a .+ randn(rng, n_obs)
    y = theta .* d .+ X[:, 1] .+ a .+ randn(rng, n_obs)
    return DoubleMLData(X, y, d; y_col="y", d_col="d", cluster=cluster, cluster_cols="cluster")
end

"""
    make_irm_data(; n_obs=500, dim_x=20, theta=0.5, seed=nothing) -> DoubleMLData

Synthetic IRM data with binary treatment and constant ATE `theta`.
"""
function make_irm_data(; n_obs::Int=500, dim_x::Int=20, theta::Real=0.5,
                       return_type::Symbol=:DoubleMLData,
                       seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    X = randn(rng, n_obs, dim_x)
    # propensity
    logits = 0.5 .* X[:, 1] .- 0.5 .* X[:, 2]
    p = 1 ./ (1 .+ exp.(-logits))
    d = Float64.(rand(rng, n_obs) .< p)
    # potential outcomes
    g0 = X[:, 1] .+ 0.5 .* X[:, 3]
    g1 = g0 .+ theta .+ 0.1 .* X[:, 2]
    y = d .* g1 .+ (1 .- d) .* g0 .+ randn(rng, n_obs)

    data = DoubleMLData(X, y, d; y_col="y", d_col="d")
    if return_type === :DataFrame
        df = DataFrame(X, :auto)
        rename!(df, ["X$i" for i in 1:dim_x])
        df.y = y
        df.d = d
        return df
    end
    return data
end

"""
    make_pliv_data(; n_obs=500, dim_x=20, dim_z=1, theta=0.5, seed=nothing)

Synthetic **partially linear IV** data (spirit of Chernozhukov–Hansen–Spindler / CHS2015).

Generates continuous `D`, continuous instruments `Z`, and outcome with true coefficient `theta`.
"""
function make_pliv_data(; n_obs::Int=500, dim_x::Int=20, dim_z::Int=1,
                        theta::Real=0.5,
                        return_type::Symbol=:DoubleMLData,
                        seed=nothing)
    dim_z >= 1 || throw(ArgumentError("dim_z must be ≥ 1"))
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    X = randn(rng, n_obs, dim_x)
    conf = X[:, 1] .+ 0.5 .* X[:, 2]
    Z = randn(rng, n_obs, dim_z)
    pi_coef = fill(1.0, dim_z)
    d = conf .+ Z * pi_coef .+ randn(rng, n_obs)
    y = theta .* d .+ conf .+ randn(rng, n_obs)

    z_cols = ["Z$i" for i in 1:dim_z]
    data = DoubleMLData(X, y, d; y_col="y", d_col="d", z=Z, z_cols=z_cols)

    if return_type === :DataFrame
        df = DataFrame(X, :auto)
        rename!(df, ["X$i" for i in 1:dim_x])
        for j in 1:dim_z
            df[!, z_cols[j]] = Z[:, j]
        end
        df.y = y
        df.d = d
        return df
    end
    return data
end

"""
    make_iivm_data(; n_obs=1000, dim_x=10, theta=0.5, seed=nothing)

Synthetic **interactive IV / LATE** data with binary treatment and binary instrument.
"""
function make_iivm_data(; n_obs::Int=1000, dim_x::Int=10, theta::Real=0.5,
                        alpha_x::Real=1.0,
                        return_type::Symbol=:DoubleMLData,
                        seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    X = randn(rng, n_obs, dim_x)
    m_logits = 0.3 .* X[:, 1]
    p_z = 1 ./ (1 .+ exp.(-m_logits))
    z = Float64.(rand(rng, n_obs) .< p_z)

    base = 0.2 .+ 0.1 .* tanh.(alpha_x .* X[:, 2])
    r0 = clamp.(base, 0.05, 0.4)
    r1 = clamp.(base .+ 0.4, 0.5, 0.95)
    d = Float64.(rand(rng, n_obs) .< (z .* r1 .+ (1 .- z) .* r0))

    g0 = X[:, 1] .+ 0.5 .* X[:, 3]
    g1 = g0 .+ theta
    y = d .* g1 .+ (1 .- d) .* g0 .+ randn(rng, n_obs)

    data = DoubleMLData(X, y, d; y_col="y", d_col="d",
                        z=reshape(z, n_obs, 1), z_cols=["z"])

    if return_type === :DataFrame
        df = DataFrame(X, :auto)
        rename!(df, ["X$i" for i in 1:dim_x])
        df.y = y
        df.d = d
        df.z = z
        return df
    end
    return data
end

"""
    make_did_data(; n_obs=500, dim_x=4, theta=-2.0, seed=nothing) -> DoubleMLData

Two-period DiD DGP in the spirit of Sant'Anna & Zhao (2020).

Returns `y = Y₁ − Y₀` (outcome change), binary treatment group `d`, and covariates.
True ATT is approximately `theta` under conditional parallel trends.
"""
function make_did_data(; n_obs::Int=500, dim_x::Int=4, theta::Real=-2.0,
                       seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dim_x >= 1 || throw(ArgumentError("dim_x ≥ 1"))
    X = randn(rng, n_obs, dim_x)
    # propensity depending on X
    logits = 0.5 .* X[:, 1] .- 0.25 .* (dim_x >= 2 ? X[:, 2] : 0.0)
    p = 1 ./ (1 .+ exp.(-logits))
    d = Float64.(rand(rng, n_obs) .< p)
    # baseline and trend
    g0 = 2 .* X[:, 1] .+ (dim_x >= 2 ? X[:, 2] : 0.0)
    # Y0, Y1 under control / treated
    y0 = g0 .+ randn(rng, n_obs)
    y1 = g0 .+ 1.0 .+ theta .* d .+ randn(rng, n_obs)  # common trend + ATT on treated
    y = y1 .- y0
    return DoubleMLData(X, y, d; y_col="y", d_col="d")
end

"""
    make_did_cs_data(; n_obs=1000, dim_x=4, theta=-2.0, seed=nothing) -> DoubleMLData

Repeated cross-section two-period DiD DGP (Chang 2020 style).
Each row is an independent unit observed at one time (`t ∈ {0,1}`).
Binary group `d`, true ATT ≈ `theta`.
"""
function make_did_cs_data(; n_obs::Int=1000, dim_x::Int=4, theta::Real=-2.0,
                         seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dim_x >= 1 || throw(ArgumentError("dim_x ≥ 1"))
    X = randn(rng, n_obs, dim_x)
    logits = 0.5 .* X[:, 1] .- 0.25 .* (dim_x >= 2 ? X[:, 2] : 0.0)
    p = 1 ./ (1 .+ exp.(-logits))
    d = Float64.(rand(rng, n_obs) .< p)
    t = Float64.(rand(rng, n_obs) .< 0.5)  # 0 pre, 1 post
    g0 = 2 .* X[:, 1] .+ (dim_x >= 2 ? X[:, 2] : 0.0)
    # level outcomes with common trend + ATT for treated in post
    y = g0 .+ 0.5 .* t .+ theta .* d .* t .+ randn(rng, n_obs)
    return DoubleMLData(X, y, d; y_col="y", d_col="d", t=Int.(t))
end

"""
    make_plpr_data(; n_id=200, n_t=4, dim_x=3, theta=0.5, seed=nothing) -> DoubleMLData

Panel DGP for partially linear panel regression (first-difference).
True slope on time-varying treatment ≈ `theta`.
"""
function make_plpr_data(; n_id::Int=200, n_t::Int=4, dim_x::Int=3,
                       theta::Real=0.5, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    n_t >= 2 || throw(ArgumentError("n_t ≥ 2"))
    id = Int[]; t = Int[]; d = Float64[]; y = Float64[]
    Xs = Vector{Vector{Float64}}()
    for i in 1:n_id
        αi = randn(rng)
        for tt in 1:n_t
            xit = randn(rng, dim_x)
            dit = 0.3 * xit[1] + randn(rng)
            yi = αi + theta * dit + 0.5 * xit[1] + 0.2 * tt + randn(rng)
            push!(Xs, xit)
            push!(id, i); push!(t, tt); push!(d, dit); push!(y, yi)
        end
    end
    X = reduce(vcat, (r' for r in Xs))
    return DoubleMLData(X, y, d; y_col="y", d_col="d", id=id, t=t)
end

"""
    make_lplr_data(; n_obs=800, dim_x=15, alpha=0.5, seed=nothing) -> DoubleMLData

Binary outcome logistic PLR DGP (Liu–Zhang–Zhou style). True slope ≈ `alpha`.
"""
function make_lplr_data(; n_obs::Int=800, dim_x::Int=15, alpha::Real=0.5, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dim_x >= 4 || throw(ArgumentError("dim_x ≥ 4"))
    X = randn(rng, n_obs, dim_x)
    a0 = 2 ./ (1 .+ exp.(X[:, 1])) .- 2 ./ (1 .+ exp.(X[:, 2])) .+ sin.(X[:, 3]) .+ cos.(X[:, 4])
    r0 = 0.1 .* X[:, 1] .* X[:, 2] .+ 0.1 .* (dim_x >= 5 ? X[:, 4] .* X[:, 5] : 0.0)
    d = a0  # continuous treatment
    p = 1 ./ (1 .+ exp.(-(alpha .* d .+ r0)))
    y = Float64.(rand(rng, n_obs) .< p)
    return DoubleMLData(X, y, d; y_col="y", d_col="d")
end

"""
    make_ssm_data(; n_obs=1000, dim_x=5, theta=1.0, seed=nothing) -> DoubleMLData

Sample selection MAR DGP. `s` is selection; `y` is observed only when `s=1`
(unobserved filled with 0 for storage).
"""
function make_ssm_data(; n_obs::Int=1000, dim_x::Int=5, theta::Real=1.0,
                       nonignorable::Bool=false, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    X = randn(rng, n_obs, dim_x)
    p_d = 1 ./ (1 .+ exp.(-0.5 .* X[:, 1]))
    d = Float64.(rand(rng, n_obs) .< p_d)
    z = nothing
    if nonignorable
        # instrument for selection (affects S, not Y directly)
        z = randn(rng, n_obs, 1)
        p_s = 1 ./ (1 .+ exp.(-(0.3 .+ 0.4 .* X[:, 1] .+ 0.3 .* d .+ 0.8 .* z[:, 1])))
    else
        p_s = 1 ./ (1 .+ exp.(-(0.5 .+ 0.5 .* X[:, 1] .+ 0.3 .* d)))
    end
    s = Float64.(rand(rng, n_obs) .< p_s)
    y_star = theta .* d .+ X[:, 1] .+ randn(rng, n_obs)
    y = ifelse.(s .== 1, y_star, 0.0)
    return DoubleMLData(X, y, d; y_col="y", d_col="d", s=s, s_col="s",
                        z=z, z_cols=z === nothing ? nothing : ["z"])
end

"""
    make_did_panel_data(; n_id=200, n_t=4, dim_x=3, theta=2.0, seed=nothing)

Staggered adoption panel (long format). `d` stores first treatment period (0=never).
"""
function make_did_panel_data(; n_id::Int=200, n_t::Int=4, dim_x::Int=3,
                             theta::Real=2.0, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    n_t >= 3 || throw(ArgumentError("n_t ≥ 3"))
    # assign groups: never, treat at 2, treat at 3, ...
    groups = [0; collect(2:n_t)]  # never + g=2..n_t
    id = Int[]; t = Int[]; d = Float64[]; y = Float64[]
    Xs = Vector{Vector{Float64}}()
    for i in 1:n_id
        g = groups[rand(rng, 1:length(groups))]
        αi = randn(rng)
        for tt in 1:n_t
            xit = randn(rng, dim_x)
            push!(Xs, xit)
            push!(id, i); push!(t, tt); push!(d, Float64(g))
            treat_now = (g > 0 && tt >= g) ? 1.0 : 0.0
            yi = αi + 0.5 * xit[1] + 0.2 * tt + theta * treat_now + randn(rng)
            push!(y, yi)
        end
    end
    X = reduce(vcat, (r' for r in Xs))
    return DoubleMLData(X, y, d; y_col="y", d_col="d", id=id, t=t)
end

"""
    make_rdd_data(; n_obs=2000, dim_x=3, tau=1.0, fuzzy=false, seed=nothing)

Simple RDD DGP with running variable `score`, cutoff 0, effect `tau`.
"""
function make_rdd_data(; n_obs::Int=2000, dim_x::Int=3, tau::Real=1.0,
                       fuzzy::Bool=false, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    score = randn(rng, n_obs)
    X = rand(rng, n_obs, dim_x) .* 2 .- 1
    g0 = 0.1 .* score .^ 2
    g1 = tau .+ 0.1 .* score .^ 2 .- 0.5 .* score
    gcov = sum(X; dims=2)[:]
    Y0 = g0 .+ gcov .+ 0.2 .* randn(rng, n_obs)
    Y1 = g1 .+ gcov .+ 0.2 .* randn(rng, n_obs)
    z = Float64.(score .>= 0)
    if fuzzy
        p = 0.1 .+ 0.8 .* z
        d = Float64.(rand(rng, n_obs) .< p)
    else
        d = z
    end
    y = (1 .- d) .* Y0 .+ d .* Y1
    return DoubleMLData(X, y, d; y_col="y", d_col="d", score=score)
end

# ---- golden-section scalar minimizer (for confounded DGPs) -----------------

function _minimize_scalar(f; lo::Float64=-20.0, hi::Float64=20.0, n_iter::Int=80)
    φ = (sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:n_iter
        if fc < fd
            b, d, fd = d, c, fc
            c = b - φ * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + φ * (b - a)
            fd = f(d)
        end
    end
    return (a + b) / 2
end

"""
    make_confounded_plr_data(; n_obs=500, theta=5.0, cf_y=0.04, cf_d=0.04,
                             dim_x=4, c=0.0, seed=nothing) -> NamedTuple

Confounded PLR DGP (Python `make_confounded_plr_data`). Returns
`(x, y, d, oracle_values, data)` where `data` is a `DoubleMLData` on observed `(X,y,d)`.
"""
function make_confounded_plr_data(; n_obs::Int=500, theta::Real=5.0,
                                  cf_y::Real=0.04, cf_d::Real=0.04,
                                  dim_x::Int=4, c::Real=0.0, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dim_x >= 4 || throw(ArgumentError("dim_x ≥ 4"))
    (0 <= cf_y < 1) || throw(ArgumentError("cf_y in [0,1)"))
    (0 <= cf_d < 1) || throw(ArgumentError("cf_d in [0,1)"))

    # Toeplitz corr
    Σ = [c^abs(i - j) for i in 1:dim_x, j in 1:dim_x]
    # simple multivariate normal via Cholesky
    L = cholesky(Symmetric(Σ + 1e-10I)).L
    x = (L * randn(rng, dim_x, n_obs))'

    z_tilde_1 = exp.(0.5 .* x[:, 1])
    z_tilde_2 = 10 .+ x[:, 2] ./ (1 .+ exp.(x[:, 1]))
    z_tilde_3 = (0.6 .+ x[:, 1] .* x[:, 3] ./ 25) .^ 3
    z_tilde_4 = (20 .+ x[:, 2] .+ x[:, 4]) .^ 2
    z_tilde = hcat(z_tilde_1, z_tilde_2, z_tilde_3, z_tilde_4, x[:, 5:end])
    z = (z_tilde .- mean(z_tilde; dims=1)) ./ std(z_tilde; dims=1)

    var_eps_y = 5.0
    eps_y = sqrt(var_eps_y) .* randn(rng, n_obs)
    var_eps_d = 1.0
    eps_d = sqrt(var_eps_d) .* randn(rng, n_obs)

    a = 2 .* rand(rng, n_obs) .- 1  # U[-1,1]
    var_a = 4.0 / 12  # (2)^2/12

    m_short = -z[:, 1] .+ 0.5 .* z[:, 2] .- 0.25 .* z[:, 3] .- 0.1 .* z[:, 4]

    function f_m(gamma_a)
        rr_long = eps_d ./ var_eps_d
        rr_short = (gamma_a .* a .+ eps_d) ./ (gamma_a^2 * var_a + var_eps_d)
        C2_D = (mean(rr_long .^ 2) - mean(rr_short .^ 2)) / mean(rr_short .^ 2)
        return (C2_D / (1 + C2_D) - cf_d)^2
    end
    gamma_a = _minimize_scalar(f_m; lo=-10.0, hi=10.0)
    m_long = m_short .+ gamma_a .* a
    d = m_long .+ eps_d

    g_partial = 210 .+ 27.4 .* z[:, 1] .+ 13.7 .* (z[:, 2] .+ z[:, 3] .+ z[:, 4])
    var_d = var(d)

    function f_g(beta_a)
        g_diff = beta_a .* (a .- gamma_a * (var_a / var_d) .* d)
        y_diff = eps_y .+ g_diff
        return (mean(g_diff .^ 2) / mean(y_diff .^ 2) - cf_y)^2
    end
    beta_a = _minimize_scalar(f_g; lo=-20.0, hi=20.0)

    g_long = theta .* d .+ g_partial .+ beta_a .* a
    g_short = (theta + gamma_a * beta_a * var_a / var_d) .* d .+ g_partial
    y = g_long .+ eps_y

    oracle = (
        g_long=g_long, g_short=g_short, m_long=m_long, m_short=m_short,
        theta=Float64(theta), gamma_a=gamma_a, beta_a=beta_a, a=a, z=z,
        cf_y=Float64(cf_y), cf_d=Float64(cf_d),
    )
    data = DoubleMLData(Matrix(x), y, d; y_col="y", d_col="d")
    return (x=Matrix(x), y=y, d=d, oracle_values=oracle, data=data)
end

"""
    make_confounded_irm_data(; n_obs=500, theta=0.0, gamma_a=0.127, beta_a=0.58,
                             linear=false, dim_x=5, seed=nothing) -> NamedTuple

Confounded IRM DGP (Python `make_confounded_irm_data`). Returns
`(x, y, d, oracle_values, data)`.
"""
function make_confounded_irm_data(; n_obs::Int=500, theta::Real=0.0,
                                  gamma_a::Real=0.127, beta_a::Real=0.58,
                                  linear::Bool=false, dim_x::Int=5,
                                  trimming_threshold::Real=0.01,
                                  var_eps_y::Real=1.0, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    dim_x >= 5 || throw(ArgumentError("dim_x ≥ 5"))
    xi = 0.75
    x = randn(rng, n_obs, dim_x)
    z_tilde = hcat(
        exp.(0.5 .* x[:, 1]),
        10 .+ x[:, 2] ./ (1 .+ exp.(x[:, 1])),
        (0.6 .+ x[:, 1] .* x[:, 3] ./ 25) .^ 3,
        (20 .+ x[:, 2] .+ x[:, 4]) .^ 2,
        x[:, 5],
    )
    z = (z_tilde .- mean(z_tilde; dims=1)) ./ std(z_tilde; dims=1)
    features = linear ? x : z

    f_ps = xi .* (-features[:, 1] .+ 0.1 .* features[:, 2] .- 0.25 .* features[:, 3] .- 0.1 .* features[:, 4])
    p = exp.(f_ps) ./ (1 .+ exp.(f_ps))
    m_long = p .+ gamma_a .* (2 .* rand(rng, n_obs) .- 1)
    # regenerate a consistently
    a = 2 .* rand(rng, n_obs) .- 1
    m_long = p .+ gamma_a .* a
    m_short = copy(p)
    thr = Float64(trimming_threshold)
    m_long = clamp.(m_long, thr, 1 - thr)
    m_short = clamp.(m_short, thr, 1 - thr)
    u = rand(rng, n_obs)
    d = Float64.(m_long .>= u)

    f_reg = 2.5 .+ 0.74 .* features[:, 1] .+ 0.25 .* features[:, 2] .+
            0.137 .* (features[:, 3] .+ features[:, 4])
    d1x = z[:, 5] .+ 1
    var_a = 4.0 / 12
    var_dx = var(d .* d1x)
    cov_adx = gamma_a * var_a
    g_short_d0 = f_reg
    g_short_d1 = (theta + beta_a * cov_adx / max(var_dx, eps())) .* d1x .+ f_reg
    g_short = d .* g_short_d1 .+ (1 .- d) .* g_short_d0
    g_long_d0 = f_reg .+ beta_a .* a
    g_long_d1 = theta .* d1x .+ f_reg .+ beta_a .* a
    g_long = d .* g_long_d1 .+ (1 .- d) .* g_long_d0
    eps_y = sqrt(var_eps_y) .* randn(rng, n_obs)
    y0 = g_long_d0 .+ eps_y
    y1 = g_long_d1 .+ eps_y
    y = d .* y1 .+ (1 .- d) .* y0

    explained = (g_long .- g_short) .^ 2
    residual = (y .- g_short) .^ 2
    cf_y_hat = mean(explained) / max(mean(residual), eps())
    cf_d_ate = (mean(1 ./ (m_long .* (1 .- m_long))) - mean(1 ./ (m_short .* (1 .- m_short)))) /
               max(mean(1 ./ (m_long .* (1 .- m_long))), eps())

    oracle = (
        g_long=g_long, g_short=g_short, m_long=m_long, m_short=m_short,
        gamma_a=Float64(gamma_a), beta_a=Float64(beta_a), a=a, y_0=y0, y_1=y1, z=z,
        cf_y=cf_y_hat, cf_d_ate=cf_d_ate, theta=Float64(theta),
    )
    # observed covariates: use z for nonlinear / x for linear (matches Python return of x)
    Xobs = linear ? x : Matrix(z)
    data = DoubleMLData(Xobs, y, d; y_col="y", d_col="d")
    return (x=Xobs, y=y, d=d, oracle_values=oracle, data=data)
end

"""
    make_heterogeneous_data(; n_obs=200, p=30, support_size=5, n_x=1,
                            binary_treatment=false, seed=nothing) -> NamedTuple

CATE DGP (Python `make_heterogeneous_data`). Returns
`(data, effects, treatment_effect, dml_data)` where `treatment_effect` is a
function of the covariate matrix and `effects` is the true CATE per row.
"""
function make_heterogeneous_data(; n_obs::Int=200, p::Int=30, support_size::Int=5,
                                 n_x::Int=1, binary_treatment::Bool=false, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    n_x in (1, 2) || throw(ArgumentError("n_x must be 1 or 2"))
    support_size <= p || throw(ArgumentError("support_size ≤ p"))

    treatment_effect = if n_x == 1
        x -> exp.(2 .* x[:, 1]) .+ 3 .* sin.(4 .* x[:, 1])
    else
        x -> exp.(2 .* x[:, 1]) .+ 3 .* sin.(4 .* x[:, 2])
    end

    X = rand(rng, n_obs, p)
    support = 1:support_size
    γ = zeros(p); β = zeros(p)
    γ[support] = rand(rng, support_size)
    β[support] = 0.3 .* rand(rng, support_size)
    η = 2 .* rand(rng, n_obs) .- 1
    ε = 2 .* rand(rng, n_obs) .- 1
    if binary_treatment
        d = Float64.((X * β) .>= η)
    else
        d = X * β .+ η
    end
    te = treatment_effect(X)
    y = te .* d .+ X * γ .+ ε
    df = DataFrame(X, :auto)
    rename!(df, ["X$i" for i in 1:p])
    df.y = y
    df.d = d
    dml = DoubleMLData(X, y, d; y_col="y", d_col="d")
    return (data=df, effects=te, treatment_effect=treatment_effect, dml_data=dml)
end

"""
    make_irm_data_discrete_treatments(; n_obs=200, n_levels=3, linear=false,
                                      seed=nothing) -> NamedTuple

Multi-level treatment IRM DGP (Python `make_irm_data_discrete_treatments`).
Returns `(x, y, d, d_cont, oracle_values, data)` with discrete `d ∈ 0:(n_levels-1)`
and continuous latent treatment `d_cont`.
"""
function make_irm_data_discrete_treatments(; n_obs::Int=200, n_levels::Int=3,
                                           linear::Bool=false, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    n_levels >= 2 || throw(ArgumentError("n_levels ≥ 2"))
    dim_x = 5
    x = randn(rng, n_obs, dim_x)
    z_tilde = hcat(
        exp.(0.5 .* x[:, 1]),
        10 .+ x[:, 2] ./ (1 .+ exp.(x[:, 1])),
        (0.6 .+ x[:, 1] .* x[:, 3] ./ 25) .^ 3,
        (20 .+ x[:, 2] .+ x[:, 4]) .^ 2,
        x[:, 5],
    )
    z = (z_tilde .- mean(z_tilde; dims=1)) ./ std(z_tilde; dims=1)
    features = linear ? x : z
    ξ = 0.3
    d_cont = ξ .* (-features[:, 1] .+ 0.5 .* features[:, 2] .- 0.25 .* features[:, 3] .- 0.1 .* features[:, 4]) .+
             randn(rng, n_obs)
    # discrete levels by quantiles (equal mass including baseline 0 for d_cont ≤ 0? Python uses quantiles of d_cont)
    # Simplified: map via quantile bins of d_cont to 0..n_levels-1
    qs = [quantile(d_cont, k / n_levels) for k in 1:(n_levels - 1)]
    d = zeros(n_obs)
    for i in 1:n_obs
        lvl = 0
        for (k, q) in enumerate(qs)
            d_cont[i] > q && (lvl = k)
        end
        d[i] = Float64(lvl)
    end
    θd = 0.1 .* exp.(d_cont) .+ 10 .* sin.(0.7 .* d_cont) .+ 2 .* d_cont .- 0.2 .* d_cont .^ 2
    y0 = 210 .+ 27.4 .* z[:, 1] .+ 13.7 .* (z[:, 2] .+ z[:, 3] .+ z[:, 4]) .+
         sqrt(5) .* randn(rng, n_obs)
    y1 = θd .* Float64.(d_cont .> 0) .+ y0
    y = ifelse.(d .> 0, y1, y0)
    Xobs = linear ? x : Matrix(z)
    data = DoubleMLData(Xobs, y, d; y_col="y", d_col="d")
    oracle = (d_cont=d_cont, theta_of_d=θd, y_0=y0, y_1=y1, z=z)
    return (x=Xobs, y=y, d=d, d_cont=d_cont, oracle_values=oracle, data=data)
end

# ---- Real datasets (Python doubleml.datasets.fetch_*) -----------------------

function _dml_cache_dir()
    d = joinpath(homedir(), ".julia", "doubleml_data")
    isdir(d) || mkpath(d)
    return d
end

"""
    fetch_401K(; return_type=:DoubleMLData, force_download=false) -> DoubleMLData | DataFrame

401(k) wealth / participation data (Abadie 2003; Chernozhukov et al. 2018).

Downloads SIPP 1991 from the DMLonGitHub repository (same URL as Python DoubleML)
and caches a CSV under `~/.julia/doubleml_data/`.

Requires network access on first call; uses Python+pandas to convert Stata if
available, otherwise errors with a clear message.
"""
function fetch_401K(; return_type::Symbol=:DoubleMLData, force_download::Bool=false)
    cache = joinpath(_dml_cache_dir(), "sipp1991.csv")
    if force_download || !isfile(cache)
        _fetch_401k_to_csv(cache)
    end
    df = _read_csv_simple(cache)
    y_col = "net_tfa"
    d_cols = ["e401"]
    x_cols = ["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]
    if return_type === :DataFrame
        return df
    end
    return DoubleMLData(df; y_col=y_col, d_cols=d_cols, x_cols=x_cols)
end

function _fetch_401k_to_csv(dest::AbstractString)
    url = "https://github.com/VC2015/DMLonGitHub/raw/master/sipp1991.dta"
    # Prefer python/pandas (matches Python DoubleML pipeline)
    py = Sys.which("python3")
    if py !== nothing
        code = """
import sys
try:
    import pandas as pd
except ImportError:
    sys.exit(2)
df = pd.read_stata($(repr(url)))
df.to_csv($(repr(dest)), index=False)
print("wrote", $(repr(dest)), "nrows", len(df))
"""
        try
            run(`$py -c $code`)
            isfile(dest) && return dest
        catch e
            @warn "python fetch_401K failed" exception=e
        end
    end
    error("fetch_401K requires network + python3 with pandas (for Stata read). " *
          "Install pandas or place a CSV at $dest with columns net_tfa, e401, age, ...")
end

"""
    fetch_bonus(; return_type=:DoubleMLData, force_download=false) -> DoubleMLData | DataFrame

Pennsylvania Reemployment Bonus experiment (Bilias 2000; Chernozhukov et al. 2018).

Same sample construction as Python: keep `tg ∈ {0,4}`, map 4→1, log `inuidur1`,
and expand `dep` into dummies.
"""
function fetch_bonus(; return_type::Symbol=:DoubleMLData, force_download::Bool=false)
    cache = joinpath(_dml_cache_dir(), "penn_jae_processed.csv")
    if force_download || !isfile(cache)
        _fetch_bonus_to_csv(cache)
    end
    df = _read_csv_simple(cache)
    y_col = "inuidur1"
    d_cols = ["tg"]
    x_cols = ["female", "black", "othrace", "dep1", "dep2", "q2", "q3", "q4", "q5", "q6",
              "agelt35", "agegt54", "durable", "lusd", "husd"]
    # keep only columns that exist
    x_cols = [c for c in x_cols if c in names(df)]
    if return_type === :DataFrame
        return df
    end
    return DoubleMLData(df; y_col=y_col, d_cols=d_cols, x_cols=x_cols)
end

function _fetch_bonus_to_csv(dest::AbstractString)
    url = "https://raw.githubusercontent.com/VC2015/DMLonGitHub/master/penn_jae.dat"
    py = Sys.which("python3")
    if py !== nothing
        code = """
import sys
try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit(2)
raw = pd.read_csv($(repr(url)), sep=r"\\s+")
ind = (raw["tg"] == 0) | (raw["tg"] == 4)
data = raw.loc[ind].copy()
data.reset_index(drop=True, inplace=True)
data["tg"] = data["tg"].replace(4, 1)
data["inuidur1"] = np.log(data["inuidur1"].clip(lower=1e-8))
# dep dummies
if "dep" in data.columns:
    data["dep1"] = (data["dep"] == 1).astype(float)
    data["dep2"] = (data["dep"] == 2).astype(float)
data.to_csv($(repr(dest)), index=False)
print("wrote", $(repr(dest)), "nrows", len(data))
"""
        try
            run(`$py -c $code`)
            isfile(dest) && return dest
        catch e
            @warn "python fetch_bonus failed" exception=e
        end
    end
    error("fetch_bonus requires network + python3 with pandas. " *
          "Place a processed CSV at $dest if offline.")
end

"""Minimal CSV reader using only DataFrames/stdlib (header + comma-separated)."""
function _read_csv_simple(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("empty CSV $path")
    header = split(strip(lines[1]), ',')
    cols = String.(header)
    data = [String[] for _ in cols]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ',')
        length(parts) == length(cols) || continue
        for (j, p) in enumerate(parts)
            push!(data[j], p)
        end
    end
    df = DataFrame()
    for (j, c) in enumerate(cols)
        vals = data[j]
        # try parse float
        parsed = Vector{Float64}(undef, length(vals))
        ok = true
        for (i, v) in enumerate(vals)
            x = tryparse(Float64, v)
            if x === nothing
                ok = false
                break
            end
            parsed[i] = x
        end
        df[!, Symbol(c)] = ok ? parsed : vals
    end
    return df
end
