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
