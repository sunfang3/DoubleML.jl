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
function make_ssm_data(; n_obs::Int=1000, dim_x::Int=5, theta::Real=1.0, seed=nothing)
    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)
    X = randn(rng, n_obs, dim_x)
    p_d = 1 ./ (1 .+ exp.(-0.5 .* X[:, 1]))
    d = Float64.(rand(rng, n_obs) .< p_d)
    # selection depends on X, D
    p_s = 1 ./ (1 .+ exp.(-(0.5 .+ 0.5 .* X[:, 1] .+ 0.3 .* d)))
    s = Float64.(rand(rng, n_obs) .< p_s)
    y_star = theta .* d .+ X[:, 1] .+ randn(rng, n_obs)
    y = ifelse.(s .== 1, y_star, 0.0)
    return DoubleMLData(X, y, d; y_col="y", d_col="d", s=s, s_col="s")
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
