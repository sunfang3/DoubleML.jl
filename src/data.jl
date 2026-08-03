"""
    DoubleMLData

Data container mirroring Python `doubleml.DoubleMLData`.

# Fields
- `x::Matrix{Float64}` — covariates (n × p)
- `y::Vector{Float64}` — outcome
- `d::Vector{Float64}` — treatment (single treatment for v0.1)
- `y_col::String`, `d_col::String`, `x_cols::Vector{String}`
"""
struct DoubleMLData
    x::Matrix{Float64}
    y::Vector{Float64}
    d::Vector{Float64}
    y_col::String
    d_col::String
    x_cols::Vector{String}
    z::Union{Nothing,Matrix{Float64}}  # instruments (optional; for PLIV later)
    z_cols::Union{Nothing,Vector{String}}
end

function DoubleMLData(x::AbstractMatrix, y::AbstractVector, d::AbstractVector;
                      y_col::AbstractString="y",
                      d_col::AbstractString="d",
                      x_cols=nothing,
                      z=nothing,
                      z_cols=nothing)
    n, p = size(x)
    length(y) == n || throw(DimensionMismatch("y length must match n_obs"))
    length(d) == n || throw(DimensionMismatch("d length must match n_obs"))
    xc = x_cols === nothing ? ["X$i" for i in 1:p] : String.(collect(x_cols))
    length(xc) == p || throw(ArgumentError("x_cols length must equal p"))
    zmat = z === nothing ? nothing : Matrix{Float64}(z)
    zc = z_cols === nothing ? nothing : String.(collect(z_cols))
    return DoubleMLData(Matrix{Float64}(x), Float64.(y), Float64.(d),
                        String(y_col), String(d_col), xc, zmat, zc)
end

"""
    DoubleMLData(df::DataFrame; y_col, d_cols, x_cols=nothing, z_cols=nothing)

Construct from a `DataFrame` (Python-compatible API).
`d_cols` may be a String or a 1-element collection (multi-treatment not yet supported).
"""
function DoubleMLData(df::DataFrame;
                      y_col::AbstractString,
                      d_cols,
                      x_cols=nothing,
                      z_cols=nothing)
    y_col = String(y_col)
    d_col = d_cols isa AbstractString ? String(d_cols) : String(first(d_cols))
    if d_cols isa AbstractVector && length(d_cols) > 1
        @warn "Multi-treatment not yet supported; using first treatment only: $d_col"
    end

    all_cols = names(df)
    string_names = String.(all_cols)

    if x_cols === nothing
        exclude = Set([y_col, d_col])
        if z_cols !== nothing
            for zc in (z_cols isa AbstractString ? [z_cols] : z_cols)
                push!(exclude, String(zc))
            end
        end
        x_cols = [c for c in string_names if c ∉ exclude]
    else
        x_cols = String.(collect(x_cols))
    end

    X = Matrix{Float64}(df[:, x_cols])
    y = Float64.(df[!, y_col])
    d = Float64.(df[!, d_col])

    z = nothing
    zc = nothing
    if z_cols !== nothing
        zc = z_cols isa AbstractString ? [String(z_cols)] : String.(collect(z_cols))
        z = Matrix{Float64}(df[:, zc])
    end

    return DoubleMLData(X, y, d; y_col=y_col, d_col=d_col, x_cols=x_cols, z=z, z_cols=zc)
end

n_obs(data::DoubleMLData) = length(data.y)
n_features(data::DoubleMLData) = size(data.x, 2)

function Base.show(io::IO, data::DoubleMLData)
    print(io, "DoubleMLData(n=$(n_obs(data)), p=$(n_features(data)), ",
          "y=$(data.y_col), d=$(data.d_col))")
end
