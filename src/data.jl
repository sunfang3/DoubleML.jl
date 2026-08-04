"""
    DoubleMLData

Data container mirroring Python `doubleml.DoubleMLData`.

# Fields
- `x` — covariates (n × p)
- `y` — outcome
- `d` — active treatment vector (column 1 of `d_mat`; backward compatible)
- `d_mat` — all treatment columns (n × n_treat)
- `d_cols` — treatment names
- `use_other_treat_as_covariate` — other Ds enter X when estimating one treatment
- `cluster` — optional n × k cluster ids (k ≤ 2)
"""
struct DoubleMLData
    x::Matrix{Float64}
    y::Vector{Float64}
    d::Vector{Float64}
    d_mat::Matrix{Float64}
    d_cols::Vector{String}
    use_other_treat_as_covariate::Bool
    y_col::String
    d_col::String
    x_cols::Vector{String}
    z::Union{Nothing,Matrix{Float64}}
    z_cols::Union{Nothing,Vector{String}}
    s::Union{Nothing,Vector{Float64}}
    s_col::Union{Nothing,String}
    id::Union{Nothing,Vector{Int}}
    t::Union{Nothing,Vector{Int}}
    score::Union{Nothing,Vector{Float64}}
    cluster::Union{Nothing,Matrix{Int}}
    cluster_cols::Union{Nothing,Vector{String}}
end

function DoubleMLData(x::AbstractMatrix, y::AbstractVector, d;
                      y_col::AbstractString="y",
                      d_col::AbstractString="d",
                      d_cols=nothing,
                      x_cols=nothing,
                      z=nothing,
                      z_cols=nothing,
                      s=nothing,
                      s_col=nothing,
                      id=nothing,
                      t=nothing,
                      score=nothing,
                      cluster=nothing,
                      cluster_cols=nothing,
                      use_other_treat_as_covariate::Bool=true)
    n, p = size(x)
    length(y) == n || throw(DimensionMismatch("y length must match n_obs"))

    d_mat = if d isa AbstractMatrix
        size(d, 1) == n || throw(DimensionMismatch("d rows must match n_obs"))
        Matrix{Float64}(d)
    else
        length(d) == n || throw(DimensionMismatch("d length must match n_obs"))
        reshape(Float64.(d), n, 1)
    end
    n_t = size(d_mat, 2)
    dc = if d_cols === nothing
        n_t == 1 ? [String(d_col)] : ["d$i" for i in 1:n_t]
    else
        String.(collect(d_cols))
    end
    length(dc) == n_t || throw(ArgumentError("d_cols length must equal n_treat=$n_t"))

    xc = x_cols === nothing ? ["X$i" for i in 1:p] : String.(collect(x_cols))
    length(xc) == p || throw(ArgumentError("x_cols length must equal p"))
    if z === nothing && z_cols !== nothing
        throw(ArgumentError("z_cols provided but z is missing"))
    end
    zmat = if z === nothing
        nothing
    elseif z isa AbstractVector
        length(z) == n || throw(DimensionMismatch("z length must match n_obs"))
        reshape(Float64.(z), n, 1)
    else
        size(z, 1) == n || throw(DimensionMismatch("z rows must match n_obs"))
        Matrix{Float64}(z)
    end
    if zmat !== nothing && size(zmat, 2) < 1
        throw(ArgumentError("z must have at least one column"))
    end
    zc = if zmat === nothing
        nothing
    elseif z_cols === nothing
        ["Z$i" for i in 1:size(zmat, 2)]
    else
        names = String.(collect(z_cols))
        length(names) == size(zmat, 2) ||
            throw(ArgumentError("z_cols length must equal n_instr=$(size(zmat, 2))"))
        names
    end
    svec = s === nothing ? nothing : Float64.(s)
    if svec !== nothing
        length(svec) == n || throw(DimensionMismatch("s length"))
    end
    idv = id === nothing ? nothing : Int.(collect(id))
    tv = t === nothing ? nothing : Int.(collect(t))
    sc = score === nothing ? nothing : Float64.(collect(score))
    idv !== nothing && length(idv) != n && throw(DimensionMismatch("id length"))
    tv !== nothing && length(tv) != n && throw(DimensionMismatch("t length"))
    sc !== nothing && length(sc) != n && throw(DimensionMismatch("score length"))

    cl, clc = _parse_cluster(cluster, cluster_cols, n)

    return DoubleMLData(
        Matrix{Float64}(x), Float64.(y),
        vec(d_mat[:, 1]), d_mat, dc, use_other_treat_as_covariate,
        String(y_col), dc[1], xc, zmat, zc,
        svec, s_col === nothing ? nothing : String(s_col),
        idv, tv, sc, cl, clc,
    )
end

function _parse_cluster(cluster, cluster_cols, n::Int)
    if cluster === nothing && cluster_cols === nothing
        return nothing, nothing
    end
    cluster === nothing && throw(ArgumentError("cluster_cols set but cluster matrix missing"))
    cl = if cluster isa AbstractVector
        length(cluster) == n || throw(DimensionMismatch("cluster length"))
        reshape(Int.(cluster), n, 1)
    else
        size(cluster, 1) == n || throw(DimensionMismatch("cluster rows"))
        size(cluster, 2) <= 2 || throw(ArgumentError("at most 2 cluster variables supported"))
        size(cluster, 2) >= 1 || throw(ArgumentError("cluster must have ≥1 column"))
        Int.(cluster)
    end
    k = size(cl, 2)
    clc = if cluster_cols === nothing
        k == 1 ? ["cluster"] : ["cluster$i" for i in 1:k]
    elseif cluster_cols isa AbstractString
        k == 1 || throw(ArgumentError("single cluster_cols name but k=$k"))
        [String(cluster_cols)]
    else
        String.(collect(cluster_cols))
    end
    length(clc) == k || throw(ArgumentError("cluster_cols length must equal n_cluster_vars=$k"))
    return cl, clc
end

"""
    DoubleMLData(df::DataFrame; y_col, d_cols, x_cols=nothing, ...)

Construct from a `DataFrame` (Python-compatible API).
`d_cols` may be a String or a collection of treatment column names.
"""
function DoubleMLData(df::DataFrame;
                      y_col::AbstractString,
                      d_cols,
                      x_cols=nothing,
                      z_cols=nothing,
                      s_col=nothing,
                      id_col=nothing,
                      t_col=nothing,
                      score_col=nothing,
                      cluster_cols=nothing,
                      use_other_treat_as_covariate::Bool=true)
    y_col = String(y_col)
    dc = d_cols isa AbstractString ? [String(d_cols)] : String.(collect(d_cols))
    length(dc) >= 1 || throw(ArgumentError("d_cols must be non-empty"))

    all_cols = names(df)
    string_names = String.(all_cols)

    clc = if cluster_cols === nothing
        nothing
    elseif cluster_cols isa AbstractString
        [String(cluster_cols)]
    else
        String.(collect(cluster_cols))
    end

    if x_cols === nothing
        exclude = Set(vcat([y_col], dc))
        if z_cols !== nothing
            for zc in (z_cols isa AbstractString ? [z_cols] : z_cols)
                push!(exclude, String(zc))
            end
        end
        s_col !== nothing && push!(exclude, String(s_col))
        id_col !== nothing && push!(exclude, String(id_col))
        t_col !== nothing && push!(exclude, String(t_col))
        score_col !== nothing && push!(exclude, String(score_col))
        if clc !== nothing
            for c in clc
                push!(exclude, c)
            end
        end
        x_cols = [c for c in string_names if c ∉ exclude]
    else
        x_cols = String.(collect(x_cols))
    end

    X = Matrix{Float64}(df[:, x_cols])
    y = Float64.(df[!, y_col])
    d_mat = Matrix{Float64}(df[:, dc])

    z = nothing
    zc = nothing
    if z_cols !== nothing
        zc = z_cols isa AbstractString ? [String(z_cols)] : String.(collect(z_cols))
        z = Matrix{Float64}(df[:, zc])
    end
    s = s_col === nothing ? nothing : Float64.(df[!, String(s_col)])
    idv = id_col === nothing ? nothing : Int.(df[!, String(id_col)])
    tv = t_col === nothing ? nothing : Int.(df[!, String(t_col)])
    scv = score_col === nothing ? nothing : Float64.(df[!, String(score_col)])
    cl = clc === nothing ? nothing : Int.(Matrix(df[:, clc]))

    return DoubleMLData(X, y, d_mat; y_col=y_col, d_cols=dc, x_cols=x_cols, z=z, z_cols=zc,
                        s=s, s_col=s_col === nothing ? nothing : String(s_col),
                        id=idv, t=tv, score=scv, cluster=cl, cluster_cols=clc,
                        use_other_treat_as_covariate=use_other_treat_as_covariate)
end

n_obs(data::DoubleMLData) = length(data.y)
n_features(data::DoubleMLData) = size(data.x, 2)
n_instr(data::DoubleMLData) = data.z === nothing ? 0 : size(data.z, 2)
n_treat(data::DoubleMLData) = size(data.d_mat, 2)
n_cluster_vars(data::DoubleMLData) = data.cluster === nothing ? 0 : size(data.cluster, 2)
is_cluster_data(data::DoubleMLData) = data.cluster !== nothing

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

"""
    design_for_treatment(data, j) -> (X, d, name)

Covariates and treatment vector for estimating treatment column `j`.
If `use_other_treat_as_covariate`, other treatment columns are appended to X.
"""
function design_for_treatment(data::DoubleMLData, j::Int)
    (1 <= j <= n_treat(data)) || throw(BoundsError(data.d_cols, j))
    d = @view data.d_mat[:, j]
    X = if data.use_other_treat_as_covariate && n_treat(data) > 1
        others_idx = [i for i in 1:n_treat(data) if i != j]
        hcat(data.x, data.d_mat[:, others_idx])
    else
        data.x
    end
    return X, d, data.d_cols[j]
end

function Base.show(io::IO, data::DoubleMLData)
    zinfo = data.z === nothing ? "none" : join(data.z_cols::Vector{String}, ",")
    dinfo = join(data.d_cols, ",")
    clinfo = data.cluster_cols === nothing ? "none" : join(data.cluster_cols, ",")
    print(io, "DoubleMLData(n=$(n_obs(data)), p=$(n_features(data)), ",
          "y=$(data.y_col), d=[$dinfo], z=$zinfo, cluster=$clinfo)")
end
