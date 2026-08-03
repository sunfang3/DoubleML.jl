# Cluster-robust standard errors (one-way)
# Python: DoubleMLClusterData / multiway cluster framework (one-way subset)

"""
    cluster_se(m; cluster, level=0.95) -> NamedTuple

One-way cluster-robust SEs for a fitted model using stored influence scores:

```
IF_i = ψ_i / J ,   SE = sqrt( Σ_c (Σ_{i∈c} IF_i)² ) / n
```

# Arguments
- `cluster`: length-`n` vector of cluster ids (e.g. firm, state)
"""
function cluster_se(m::AbstractDoubleML; cluster::AbstractVector, level::Real=0.95)
    m.fitted || error("Call fit! first")
    n = size(m.psi, 1)
    length(cluster) == n || throw(DimensionMismatch("cluster length must equal n"))
    n_coef = length(m.coef)
    n_rep = size(m.psi, 2)
    ses = zeros(n_coef)
    for j in 1:n_coef
        # average IF across reps
        IF = zeros(n)
        for r in 1:n_rep
            J = mean(@view m.psi_deriv[:, r, j])
            abs(J) < 1e-14 && error("Degenerate J")
            IF .+= @view(m.psi[:, r, j]) ./ J
        end
        IF ./= n_rep
        # sum IF within clusters
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
