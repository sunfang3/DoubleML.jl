"""
K-fold partition for cross-fitting.

Returns a vector of `(train_idx, test_idx)` named tuples covering `1:n`.
"""
function make_folds(n::Int, n_folds::Int; rng::AbstractRNG=Random.default_rng(),
                    shuffle_rows::Bool=true)
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))
    n >= n_folds || throw(ArgumentError("n must be ≥ n_folds"))

    idx = collect(1:n)
    if shuffle_rows
        idx = Random.shuffle(rng, idx)
    end

    # balanced fold sizes
    fold_sizes = fill(n ÷ n_folds, n_folds)
    for i in 1:(n % n_folds)
        fold_sizes[i] += 1
    end

    folds = Vector{NamedTuple{(:train, :test),Tuple{Vector{Int},Vector{Int}}}}(undef, n_folds)
    start = 1
    for k in 1:n_folds
        stop = start + fold_sizes[k] - 1
        test = sort(idx[start:stop])
        train = sort(vcat(idx[1:start-1], idx[stop+1:end]))
        folds[k] = (train=train, test=test)
        start = stop + 1
    end
    return folds
end

"""
Repeated sample splitting: `n_rep` independent K-fold partitions.
"""
function make_repeated_folds(n::Int, n_folds::Int, n_rep::Int;
                             rng::AbstractRNG=Random.default_rng())
    return [make_folds(n, n_folds; rng=rng) for _ in 1:n_rep]
end

"""
    make_cluster_repeated_folds(n, n_folds, n_rep, cluster; rng)

Cluster-aware repeated sample splitting (Python `DoubleMLClusterResampling`).

For each cluster variable, unique cluster ids are K-fold split. Observation-level
folds are the Cartesian product across cluster variables (so clusters are never
split across train/test).

# Returns
Vector of length `n_rep`, each entry:
```
(folds = Vector{(train,test)},
 cluster_folds = Vector{(train_ids, test_ids)})
```
where `train_ids[v]` / `test_ids[v]` are vectors of cluster ids for variable `v`.
"""
function make_cluster_repeated_folds(n::Int, n_folds::Int, n_rep::Int,
                                     cluster::AbstractMatrix{<:Integer};
                                     rng::AbstractRNG=Random.default_rng())
    size(cluster, 1) == n || throw(DimensionMismatch("cluster rows must equal n"))
    n_cv = size(cluster, 2)
    (1 <= n_cv <= 2) || throw(ArgumentError("n_cluster_vars must be 1 or 2"))
    n_folds >= 2 || throw(ArgumentError("n_folds must be ≥ 2"))

    results = Vector{NamedTuple}(undef, n_rep)
    for r in 1:n_rep
        # per cluster-var: list of (train_cluster_ids, test_cluster_ids) length n_folds
        smpls_cluster_vars = Vector{Vector{NTuple{2,Vector{Int}}}}(undef, n_cv)
        for v in 1:n_cv
            ids = sort(unique(@view cluster[:, v]))
            n_cl = length(ids)
            n_cl >= n_folds || throw(ArgumentError(
                "cluster var $v has $n_cl clusters < n_folds=$n_folds"))
            # fold over cluster indices 1:n_cl
            cfolds = make_folds(n_cl, n_folds; rng=rng)
            smpls_cluster_vars[v] = [
                (ids[cf.train], ids[cf.test]) for cf in cfolds
            ]
        end

        # Cartesian product of fold indices
        cart = _fold_cartesian(n_folds, n_cv)
        folds = Vector{NamedTuple{(:train, :test),Tuple{Vector{Int},Vector{Int}}}}(undef, size(cart, 1))
        cluster_folds = Vector{NamedTuple}(undef, size(cart, 1))
        for i in 1:size(cart, 1)
            train_mask = trues(n)
            test_mask = trues(n)
            train_ids = Vector{Vector{Int}}(undef, n_cv)
            test_ids = Vector{Vector{Int}}(undef, n_cv)
            for v in 1:n_cv
                i_fold = cart[i, v]
                tr_c, te_c = smpls_cluster_vars[v][i_fold]
                train_ids[v] = tr_c
                test_ids[v] = te_c
                train_mask .&= in.(cluster[:, v], Ref(Set(tr_c)))
                test_mask .&= in.(cluster[:, v], Ref(Set(te_c)))
            end
            folds[i] = (train=findall(train_mask), test=findall(test_mask))
            cluster_folds[i] = (train_ids=train_ids, test_ids=test_ids)
        end
        results[r] = (folds=folds, cluster_folds=cluster_folds)
    end
    return results
end

function _fold_cartesian(n_folds::Int, n_cv::Int)
    # rows: all combinations of fold indices (1:n_folds)^n_cv
    if n_cv == 1
        return reshape(collect(1:n_folds), n_folds, 1)
    end
    rows = n_folds^n_cv
    cart = Matrix{Int}(undef, rows, n_cv)
    i = 1
    for f1 in 1:n_folds, f2 in 1:n_folds
        cart[i, 1] = f1
        cart[i, 2] = f2
        i += 1
    end
    return cart
end

"""
    init_sample_splitting(data, n_folds, n_rep; rng)

Draw observation folds; if data has clusters, use cluster-aware splitting.
Returns `(smpls, smpls_cluster, n_folds_effective)` where `smpls_cluster` may be `nothing`.
"""
function init_sample_splitting(data::DoubleMLData, n_folds::Int, n_rep::Int;
                               rng::AbstractRNG=Random.default_rng())
    n = n_obs(data)
    if is_cluster_data(data)
        raw = make_cluster_repeated_folds(n, n_folds, n_rep, data.cluster; rng=rng)
        smpls = [r.folds for r in raw]
        smpls_cluster = [r.cluster_folds for r in raw]
        return smpls, smpls_cluster, n_folds
    else
        return make_repeated_folds(n, n_folds, n_rep; rng=rng), nothing, n_folds
    end
end
