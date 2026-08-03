# Policy tree for optimal treatment assignment (IRM)
# Mirrors Python doubleml.utils.policytree.DoubleMLPolicyTree
#
# Objective (Athey & Wager / DoubleML):
#   π̂ = argmax_{π ∈ Π} (1/n) Σ_i (2π(X_i) − 1) ψ̂_b(W_i, η̂)
# which is equivalent to weighted classification of sign(ψ̂_b) with weights |ψ̂_b|.

"""
    DoubleMLPolicyTree

Depth-limited decision tree for a binary treatment policy, fit by weighted
classification of the IRM orthogonal score `ψ_b`.

Construct via [`policy_tree`](@ref) on a fitted [`DoubleMLIRM`](@ref).
"""
mutable struct DoubleMLPolicyTree
    orth_signal::Vector{Float64}
    features::Matrix{Float64}
    feature_names::Vector{String}
    depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    tree::Any                 # DecisionTree Root / Node / Leaf
    fitted::Bool
    rng::AbstractRNG
end

function Base.show(io::IO, m::DoubleMLPolicyTree)
    if !m.fitted
        print(io, "DoubleMLPolicyTree(not fitted; depth=$(m.depth), p=$(length(m.feature_names)))")
    else
        v = policy_value(m)
        print(io, "DoubleMLPolicyTree(depth=$(m.depth), policy_value=$(round(v; digits=4)), ",
              "features=$(m.feature_names))")
    end
end

"""
    policy_value(orth_signal, treatment) -> Float64
    policy_value(pt::DoubleMLPolicyTree; features=nothing) -> Float64

Estimated value of a binary policy under the IRM score:

    (1/n) Σ_i (2 π_i − 1) ψ̂_b,i

If `features` is given, predictions are formed on that design; otherwise the
training features / in-sample predictions are used.
"""
function policy_value(orth_signal::AbstractVector, treatment::AbstractVector)
    length(orth_signal) == length(treatment) ||
        throw(DimensionMismatch("orth_signal and treatment length differ"))
    π = Float64.(treatment)
    return mean((2 .* π .- 1) .* Float64.(orth_signal))
end

function policy_value(m::DoubleMLPolicyTree; features=nothing)
    m.fitted || error("Call fit! first")
    X = features === nothing ? m.features : _policy_features_matrix(features, m.feature_names)[1]
    π = predict_policy(m, X)
    return policy_value(m.orth_signal, π)
end

# ---- feature helpers --------------------------------------------------------

function _policy_features_matrix(features::AbstractMatrix, names_hint::Union{Nothing,Vector{String}}=nothing)
    X = Matrix{Float64}(features)
    col_names = if names_hint !== nothing && length(names_hint) == size(X, 2)
        names_hint
    else
        ["X$i" for i in 1:size(X, 2)]
    end
    return X, col_names
end

function _policy_features_matrix(features::DataFrame, names_hint::Union{Nothing,Vector{String}}=nothing)
    X = Matrix{Float64}(features)
    # use propertynames — avoid shadowing Base/DataFrames.names
    col_names = string.(propertynames(features))
    return X, col_names
end

# ---- weighted CART via DecisionTree.jl --------------------------------------

"""
Build a classification tree with sample weights (DecisionTree.jl backend).

`build_tree` does not expose weights; we call the weighted `treeclassifier.fit`
and wrap the result with `_build_tree`, same path as `build_stump`.
"""
function _weighted_class_tree(y::AbstractVector{<:Integer},
                              X::AbstractMatrix,
                              w::AbstractVector;
                              max_depth::Int=2,
                              min_samples_leaf::Int=8,
                              min_samples_split::Int=2,
                              min_purity_increase::Float64=0.0,
                              rng::AbstractRNG=Random.default_rng())
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("y and X size mismatch"))
    length(w) == n || throw(DimensionMismatch("weights and X size mismatch"))
    max_depth >= 1 || throw(ArgumentError("max_depth must be ≥ 1"))
    min_samples_leaf >= 1 || throw(ArgumentError("min_samples_leaf must be ≥ 1"))

    Xf = Matrix{Float64}(X)
    y_int = Int.(y)
    # DecisionTree requires strictly positive weights for numerical stability
    w_pos = max.(Float64.(w), eps(Float64))

    raw = DecisionTree.treeclassifier.fit(;
        X = Xf,
        Y = y_int,
        W = w_pos,
        loss = DecisionTree.util.entropy,
        max_features = p,
        max_depth = Int(max_depth),
        min_samples_leaf = Int(min_samples_leaf),
        min_samples_split = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng = rng,
    )
    return DecisionTree._build_tree(raw, y_int, p, n, false)
end

# ---- fit / predict ----------------------------------------------------------

"""
    fit!(pt::DoubleMLPolicyTree) -> DoubleMLPolicyTree

Fit the policy tree: classify `sign(ψ_b)` with sample weights `|ψ_b|`.
"""
function fit!(m::DoubleMLPolicyTree)
    s = m.orth_signal
    # labels: 1 = treat, 0 = do not treat  (Python: (sign(s)+1)/2)
    y = Int.(s .> 0)
    w = abs.(s)
    # if all scores zero, assign never-treat stump
    if all(iszero, w)
        @warn "Orthogonal signal is identically zero; policy is constantly 0"
        y = zeros(Int, length(s))
        w = ones(length(s))
    end
    m.tree = _weighted_class_tree(
        y, m.features, w;
        max_depth = m.depth,
        min_samples_leaf = m.min_samples_leaf,
        min_samples_split = m.min_samples_split,
        min_purity_increase = m.min_purity_increase,
        rng = m.rng,
    )
    m.fitted = true
    return m
end

"""
    predict_policy(pt, features) -> Vector{Int}

Predicted treatment assignment in `{0,1}` for each row of `features`.
"""
function predict_policy(m::DoubleMLPolicyTree, features::AbstractMatrix)
    m.fitted || error("Call fit! before predict_policy")
    X = Matrix{Float64}(features)
    size(X, 2) == size(m.features, 2) ||
        throw(DimensionMismatch("expected $(size(m.features, 2)) features, got $(size(X, 2))"))
    return Int.(DecisionTree.apply_tree(m.tree, X))
end

function predict_policy(m::DoubleMLPolicyTree, features::DataFrame)
    X, nm = _policy_features_matrix(features)
    # column order must match training names if possible
    if nm != m.feature_names
        # try to reorder
        idx = indexin(m.feature_names, nm)
        any(isnothing, idx) &&
            throw(ArgumentError("features columns $(nm) do not match training $(m.feature_names)"))
        X = X[:, Int.(idx)]
    end
    return predict_policy(m, X)
end

"""
Alias matching learner API: `predict(pt, X)` → treatment in {0,1}.
"""
predict(m::DoubleMLPolicyTree, features::AbstractVecOrMat) = predict_policy(m, features)
predict(m::DoubleMLPolicyTree, features::DataFrame) = predict_policy(m, features)

"""
    print_policy_tree(pt; kwargs...)

Pretty-print the fitted tree (wraps `DecisionTree.print_tree`).
Feature indices are 1-based as in DecisionTree.jl; see `pt.feature_names`.
"""
function print_policy_tree(m::DoubleMLPolicyTree; kwargs...)
    m.fitted || error("Call fit! first")
    println("Feature names (1-based index):")
    for (i, n) in enumerate(m.feature_names)
        println("  $i → $n")
    end
    println("Classes: 0 = No Treatment, 1 = Treatment")
    DecisionTree.print_tree(m.tree; kwargs...)
    return nothing
end

function summary_table(m::DoubleMLPolicyTree)
    m.fitted || error("Call fit! first")
    π = predict_policy(m, m.features)
    n = length(π)
    n_treat = sum(π)
    return DataFrame(
        depth = [m.depth],
        n_features = [length(m.feature_names)],
        n_obs = [n],
        n_treated = [n_treat],
        share_treated = [n_treat / n],
        policy_value = [policy_value(m.orth_signal, π)],
        features = [join(m.feature_names, ", ")],
    )
end

# ---- public entry point -----------------------------------------------------

"""
    policy_tree(m::DoubleMLIRM, features; depth=2, min_samples_leaf=8, ...) -> DoubleMLPolicyTree

Learn a depth-`depth` treatment policy from a fitted IRM model (ATE score).

# Arguments
- `m`: fitted [`DoubleMLIRM`](@ref) with `score="ATE"`.
- `features`: `n × d` matrix or `DataFrame` of covariates the policy may use
  (original `X`, a subset, or new features). Must have `n` rows matching `m`.
- `depth`: maximum tree depth (default `2`, as in Python DoubleML).
- `min_samples_leaf`: default `8` (Python default).
- `min_samples_split`: default `2`.
- `min_purity_increase`: default `0.0`.
- `rng`: RNG for any tie-breaking in the tree.

# Returns
Fitted [`DoubleMLPolicyTree`](@ref). Use [`predict_policy`](@ref),
[`policy_value`](@ref), [`print_policy_tree`](@ref), [`summary_table`](@ref).

# Notes
- Uses the IRM orthogonal score `ψ_b` (doubly robust individual effect signal).
- If `n_rep > 1`, scores are averaged across repetitions.
- Mirrors Python `DoubleMLIRM.policy_tree`.
"""
function policy_tree(m::DoubleMLIRM, features;
                     depth::Int=2,
                     min_samples_leaf::Int=8,
                     min_samples_split::Int=2,
                     min_purity_increase::Real=0.0,
                     rng::AbstractRNG=Random.default_rng())
    m.fitted || error("Call fit! before policy_tree")
    m.score == "ATE" ||
        throw(ArgumentError("policy_tree requires IRM score=\"ATE\" (got $(m.score))"))

    S = _orth_signal_irm(m)
    # average over cross-fitting repetitions (Python only allows n_rep=1)
    s = vec(mean(S; dims=2))

    X, names = _policy_features_matrix(features)
    size(X, 1) == length(s) ||
        throw(DimensionMismatch("features has $(size(X, 1)) rows but model has $(length(s)) observations"))

    pt = DoubleMLPolicyTree(
        s, X, names, depth,
        min_samples_leaf, min_samples_split, Float64(min_purity_increase),
        nothing, false, rng,
    )
    return fit!(pt)
end
