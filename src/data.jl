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
    z::Union{Nothing,Matrix{Float64}}  # instruments (optional)
    z_cols::Union{Nothing,Vector{String}}
    s::Union{Nothing,Vector{Float64}}  # selection indicator (SSM)
    s_col::Union{Nothing,String}
    # panel / RDD extras stored loosely
    id::Union{Nothing,Vector{Int}}
    t::Union{Nothing,Vector{Int}}
    score::Union{Nothing,Vector{Float64}}  # running variable for RDD
end

function DoubleMLData(x::AbstractMatrix, y::AbstractVector, d::AbstractVector;
                      y_col::AbstractString="y",
                      d_col::AbstractString="d",
                      x_cols=nothing,
                      z=nothing,
                      z_cols=nothing,
                      s=nothing,
                      s_col=nothing,
                      id=nothing,
                      t=nothing,
                      score=nothing)
    n, p = size(x)
    length(y) == n || throw(DimensionMismatch("y length must match n_obs"))
    length(d) == n || throw(DimensionMismatch("d length must match n_obs"))
    xc = x_cols === nothing ? ["X$i" for i in 1:p] : String.(collect(x_cols))
    length(xc) == p || throw(ArgumentError("x_cols length must equal p"))
    zmat = z === nothing ? nothing : Matrix{Float64}(z)
    zc = z_cols === nothing ? nothing : String.(collect(z_cols))
    svec = s === nothing ? nothing : Float64.(s)
    if svec !== nothing
        length(svec) == n || throw(DimensionMismatch("s length"))
    end
    idv = id === nothing ? nothing : Int.(id)
    tv = t === nothing ? nothing : Int.(t)
    sc = score === nothing ? nothing : Float64.(score)
    return DoubleMLData(Matrix{Float64}(x), Float64.(y), Float64.(d),
                        String(y_col), String(d_col), xc, zmat, zc,
                        svec, s_col === nothing ? nothing : String(s_col),
                        idv, tv, sc)
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
                      z_cols=nothing,
                      s_col=nothing,
                      id_col=nothing,
                      t_col=nothing,
                      score_col=nothing)
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
        s_col !== nothing && push!(exclude, String(s_col))
        id_col !== nothing && push!(exclude, String(id_col))
        t_col !== nothing && push!(exclude, String(t_col))
        score_col !== nothing && push!(exclude, String(score_col))
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
    s = s_col === nothing ? nothing : Float64.(df[!, String(s_col)])
    idv = id_col === nothing ? nothing : Int.(df[!, String(id_col)])
    tv = t_col === nothing ? nothing : Int.(df[!, String(t_col)])
    sc = score_col === nothing ? nothing : Float64.(df[!, String(score_col)])

    return DoubleMLData(X, y, d; y_col=y_col, d_col=d_col, x_cols=x_cols, z=z, z_cols=zc,
                        s=s, s_col=s_col === nothing ? nothing : String(s_col),
                        id=idv, t=tv, score=sc)
end

n_obs(data::DoubleMLData) = length(data.y)
n_features(data::DoubleMLData) = size(data.x, 2)
n_instr(data::DoubleMLData) = data.z === nothing ? 0 : size(data.z, 2)

"""Return instrument matrix as `n × k`, or throw if absent."""
function instruments(data::DoubleMLData)
    data.z === nothing && error("No instruments in DoubleMLData")
    return data.z
end

"""Single instrument as a vector (errors if not exactly one)."""
function instrument(data::DoubleMLData)
    Z = instruments(data)
    size(Z, 2) == 1 || error("Expected a single instrument, got $(size(Z, 2))")
    return vec(Z)
end

function Base.show(io::IO, data::DoubleMLData)
    zinfo = data.z === nothing ? "none" : join(data.z_cols, ",")
    print(io, "DoubleMLData(n=$(n_obs(data)), p=$(n_features(data)), ",
          "y=$(data.y_col), d=$(data.d_col), z=$zinfo)")
end
