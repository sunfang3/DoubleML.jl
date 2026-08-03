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
