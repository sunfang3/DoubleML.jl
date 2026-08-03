# Cluster-robust standard errors
# - Post-hoc one-way: cluster_se / apply_cluster_se!
# - In-fit 1/2-way (Python _var_est): var_est

"""
    var_est(psi, psi_deriv, smpls; cluster, smpls_cluster, n_folds_per_cluster)

Variance estimator matching Python `doubleml.utils._estimation._var_est`.

# Returns
`(sigma2, var_scaling_factor)` so `se = sqrt(sigma2)`.
"""
function var_est(psi::AbstractVector, psi_deriv::AbstractVector, smpls;
                 cluster::Union{Nothing,AbstractMatrix}=nothing,
                 smpls_cluster=nothing,
                 n_folds_per_cluster::Union{Nothing,Int}=nothing)
    if cluster === nothing
        n = length(psi)
        J = mean(psi_deriv)
        γ = mean(psi .^ 2)
        σ2 = γ / (J^2) / n
        return σ2, Float64(n)
    end

    smpls_cluster === nothing && error("smpls_cluster required for cluster var_est")
    n_folds_per_cluster === nothing && error("n_folds_per_cluster required")
    n_folds = length(smpls)
    n_cv = size(cluster, 2)

    if n_cv == 1
        first_cv = @view cluster[:, 1]
        clusters = unique(first_cv)
        γ = 0.0
        j_hat = 0.0
        for i_fold in 1:n_folds
            test_inds = smpls[i_fold].test
            I_k = smpls_cluster[i_fold].test_ids[1]
            const_ = 1 / length(I_k)
            for cval in I_k
                ind = first_cv .== cval
                # only observations in this cluster (and typically in test)
                ψc = psi[ind]
                γ += const_ * sum(ψc)^2   # sum of outer product diagonal form: 1'ψψ'1 = (sum ψ)^2
            end
            j_hat += sum(@view psi_deriv[test_inds]) / length(I_k)
        end
        var_scaling = Float64(length(clusters))
        J = j_hat / n_folds_per_cluster
        γ /= n_folds_per_cluster
        σ2 = γ / (var_scaling * J^2)
        return σ2, var_scaling
    elseif n_cv == 2
        first_cv = @view cluster[:, 1]
        second_cv = @view cluster[:, 2]
        γ = 0.0
        j_hat = 0.0
        for i_fold in 1:n_folds
            test_inds = smpls[i_fold].test
            I_k = smpls_cluster[i_fold].test_ids[1]
            J_l = smpls_cluster[i_fold].test_ids[2]
            const_ = min(length(I_k), length(J_l)) / (length(I_k) * length(J_l))^2
            J_l_set = Set(J_l)
            I_k_set = Set(I_k)
            for cval in I_k
                ind = (first_cv .== cval) .& in.(second_cv, Ref(J_l_set))
                ψc = psi[ind]
                γ += const_ * sum(ψc)^2
            end
            for cval in J_l
                ind = (second_cv .== cval) .& in.(first_cv, Ref(I_k_set))
                ψc = psi[ind]
                γ += const_ * sum(ψc)^2
            end
            j_hat += sum(@view psi_deriv[test_inds]) / (length(I_k) * length(J_l))
        end
        n1 = length(unique(first_cv))
        n2 = length(unique(second_cv))
        var_scaling = Float64(min(n1, n2))
        J = j_hat / (n_folds_per_cluster^2)
        γ /= n_folds_per_cluster^2
        σ2 = γ / (var_scaling * J^2)
        return σ2, var_scaling
    else
        throw(ArgumentError("n_cluster_vars must be 1 or 2"))
    end
end

"""
    se_from_score(psi_a, psi_b, θ; smpls, cluster, smpls_cluster, n_folds_per_cluster)

Standard error for linear DML score, iid or cluster.
Returns `(se, var_scaling_factor)`.
"""
function se_from_score(psi_a::AbstractVector, psi_b::AbstractVector, θ::Real;
                       smpls=nothing,
                       cluster::Union{Nothing,AbstractMatrix}=nothing,
                       smpls_cluster=nothing,
                       n_folds_per_cluster::Union{Nothing,Int}=nothing)
    ψ = psi_a .* θ .+ psi_b
    if cluster === nothing
        se = se_linear(psi_a, psi_b, θ)
        return se, Float64(length(ψ))
    end
    σ2, vsf = var_est(ψ, psi_a, smpls;
                      cluster=cluster, smpls_cluster=smpls_cluster,
                      n_folds_per_cluster=n_folds_per_cluster)
    return sqrt(max(σ2, 0.0)), vsf
end

"""
    cluster_se(m; cluster, level=0.95) -> NamedTuple

One-way cluster-robust SEs for a fitted model using stored influence scores (post-hoc).
"""
function cluster_se(m::AbstractDoubleML; cluster::AbstractVector, level::Real=0.95)
    m.fitted || error("Call fit! first")
    n = size(m.psi, 1)
    length(cluster) == n || throw(DimensionMismatch("cluster length must equal n"))
    n_coef = length(m.coef)
    n_rep = size(m.psi, 2)
    ses = zeros(n_coef)
    for j in 1:n_coef
        IF = zeros(n)
        for r in 1:n_rep
            J = mean(@view m.psi_deriv[:, r, j])
            abs(J) < 1e-14 && error("Degenerate J")
            IF .+= @view(m.psi[:, r, j]) ./ J
        end
        IF ./= n_rep
        clusters = unique(cluster)
        s2 = 0.0
        for c in clusters
            idx = findall(==(c), cluster)
            sc = sum(@view IF[idx])
            s2 += sc^2
        end
        ses[j] = sqrt(s2) / n
    end
    z = quantile(Normal(), 1 - (1 - level) / 2)
    return (
        coef = m.coef,
        se = ses,
        t = m.coef ./ ses,
        pvalue = 2 .* cdf.(Normal(), -abs.(m.coef ./ ses)),
        ci_lower = m.coef .- z .* ses,
        ci_upper = m.coef .+ z .* ses,
        n_clusters = length(unique(cluster)),
    )
end

"""
    apply_cluster_se!(m; cluster)

Replace `m.se` with one-way cluster-robust SEs (and scale `all_se` for n_rep=1).
"""
function apply_cluster_se!(m::AbstractDoubleML; cluster::AbstractVector)
    r = cluster_se(m; cluster=cluster)
    m.se = collect(r.se)
    if size(m.all_se, 2) == 1
        m.all_se[:, 1] = m.se
    end
    return m
end
