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
