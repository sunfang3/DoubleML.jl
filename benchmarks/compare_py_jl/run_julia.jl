#!/usr/bin/env julia
# Load shared CSVs + folds from Python run; fit Julia DoubleML; write comparison JSON.

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using DoubleML
using DataFrames
using CSV
using JSON
using Random
using Statistics
using LinearAlgebra

const OUT = joinpath(@__DIR__, "data")
const N_FOLDS = 5

function load_smpls(path)
    raw = JSON.parsefile(path)
    folds = [
        (train=Int.(f["train"]) .+ 1, test=Int.(f["test"]) .+ 1)  # Python 0-based → Julia 1-based
        for f in raw
    ]
    return [folds]  # n_rep = 1
end

function ols()
    LinearRegressionLearner()
end

function logit()
    # small L2; Python uses C=1e6 ≈ α≈0 for large n scale differs
    LogisticRegressionLearner(α=1e-6, max_iter=200)
end

function fit_plr(csv, smpls_path)
    df = CSV.read(csv, DataFrame)
    data = DoubleMLData(df; y_col="y", d_cols="d")
    m = DoubleMLPLR(data, ols(), ols(); n_folds=N_FOLDS, n_rep=1, draw_sample_splitting=false)
    set_sample_splitting!(m, load_smpls(smpls_path))
    fit!(m)
    return m
end

function main()
    results = Dict{String,Any}("julia_doubleml" => "1.1.0")

    # PLR
    m = fit_plr(joinpath(OUT, "plr.csv"), joinpath(OUT, "plr_smpls.json"))
    results["PLR"] = Dict(
        "coef" => collect(m.coef),
        "se" => collect(m.se),
        "t" => collect(m.coef ./ m.se),
        "names" => collect(m.treat_names),
    )
    f = construct_framework(m)
    results["PLR_framework"] = Dict("thetas" => collect(f.thetas), "ses" => collect(f.ses))
    f2 = 2 * f
    results["PLR_framework_2x"] = Dict("thetas" => collect(f2.thetas), "ses" => collect(f2.ses))

    # IRM
    df = CSV.read(joinpath(OUT, "irm.csv"), DataFrame)
    dcol = "d"
    ycol = "y"
    data = DoubleMLData(df; y_col=ycol, d_cols=dcol)
    m = DoubleMLIRM(data, ols(), logit(); n_folds=N_FOLDS, n_rep=1,
                    draw_sample_splitting=false, trimming_threshold=0.01)
    set_sample_splitting!(m, load_smpls(joinpath(OUT, "irm_smpls.json")))
    fit!(m)
    results["IRM"] = Dict("coef" => collect(m.coef), "se" => collect(m.se), "names" => collect(m.treat_names))

    # PLIV
    df = CSV.read(joinpath(OUT, "pliv.csv"), DataFrame)
    zcols = [c for c in names(df) if startswith(String(c), "Z")]
    data = DoubleMLData(df; y_col="y", d_cols="d", z_cols=zcols)
    ml = ols()
    m = DoubleMLPLIV(data, clone(ml), clone(ml), clone(ml); n_folds=N_FOLDS, n_rep=1,
                     draw_sample_splitting=false)
    set_sample_splitting!(m, load_smpls(joinpath(OUT, "pliv_smpls.json")))
    fit!(m)
    results["PLIV"] = Dict("coef" => collect(m.coef), "se" => collect(m.se), "names" => collect(m.treat_names))

    # IIVM
    df = CSV.read(joinpath(OUT, "iivm.csv"), DataFrame)
    zcols = [c for c in names(df) if startswith(lowercase(String(c)), "z")]
    data = DoubleMLData(df; y_col="y", d_cols="d", z_cols=zcols)
    m = DoubleMLIIVM(data, ols(), logit(), logit(); n_folds=N_FOLDS, n_rep=1,
                     draw_sample_splitting=false, trimming_threshold=0.05)
    set_sample_splitting!(m, load_smpls(joinpath(OUT, "iivm_smpls.json")))
    fit!(m)
    results["IIVM"] = Dict("coef" => collect(m.coef), "se" => collect(m.se), "names" => collect(m.treat_names))

    # Multi PLR
    df = CSV.read(joinpath(OUT, "plr_multi.csv"), DataFrame)
    data = DoubleMLData(df; y_col="y", d_cols=["d1", "d2"], use_other_treat_as_covariate=true)
    m = DoubleMLPLR(data, ols(), ols(); n_folds=N_FOLDS, n_rep=1, draw_sample_splitting=false)
    set_sample_splitting!(m, load_smpls(joinpath(OUT, "plr_multi_smpls.json")))
    fit!(m)
    results["PLR_multi"] = Dict("coef" => collect(m.coef), "se" => collect(m.se), "names" => collect(m.treat_names))

    # Cluster PLR (cluster-in-fit, n_folds=3 to match Python script)
    df = CSV.read(joinpath(OUT, "plr_cluster.csv"), DataFrame)
    data = DoubleMLData(df; y_col="y", d_cols="d", cluster_cols="cluster")
    m = DoubleMLPLR(data, ols(), ols(); n_folds=3, n_rep=1, rng=MersenneTwister(3141 + 5))
    fit!(m)
    results["PLR_cluster"] = Dict(
        "coef" => collect(m.coef),
        "se" => collect(m.se),
        "is_cluster_data" => m.is_cluster_data,
        "var_scaling" => m.var_scaling === nothing ? nothing : collect(m.var_scaling),
        "note" => "cluster folds RNG not synced with Python",
    )

    # APO contrast on same IRM data + folds
    df = CSV.read(joinpath(OUT, "irm.csv"), DataFrame)
    data = DoubleMLData(df; y_col="y", d_cols="d")
    smpls = load_smpls(joinpath(OUT, "irm_smpls.json"))
    apo0 = DoubleMLAPO(data, ols(), logit(); treatment_level=0.0, n_folds=N_FOLDS, n_rep=1,
                       draw_sample_splitting=false, trimming_threshold=0.01)
    apo1 = DoubleMLAPO(data, ols(), logit(); treatment_level=1.0, n_folds=N_FOLDS, n_rep=1,
                       draw_sample_splitting=false, trimming_threshold=0.01)
    set_sample_splitting!(apo0, smpls)
    set_sample_splitting!(apo1, smpls)
    fit!(apo0); fit!(apo1)
    f0 = construct_framework(apo0)
    f1 = construct_framework(apo1)
    ate = f1 - f0
    results["APO_contrast"] = Dict(
        "apo0" => collect(f0.thetas),
        "apo1" => collect(f1.thetas),
        "ate" => collect(ate.thetas),
        "ate_se" => collect(ate.ses),
        "irm_ate" => results["IRM"]["coef"],
    )

    out = joinpath(OUT, "julia_results.json")
    open(out, "w") do io
        JSON.print(io, results, 2)
    end
    println(JSON.json(results, 2))
    println("wrote ", out)
end

main()
