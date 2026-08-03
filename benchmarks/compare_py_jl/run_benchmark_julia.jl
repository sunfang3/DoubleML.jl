#!/usr/bin/env julia
# Full algorithm benchmark: Julia DoubleML.jl side.
# Loads CSVs/folds from Python run; writes data/benchmark_jl.json
using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
# temporary deps for CSV/JSON if missing
try
    @eval using CSV, JSON, DataFrames
catch
    Pkg.add(["CSV", "JSON"])
    @eval using CSV, JSON, DataFrames
end

using DoubleML
using Random
using Statistics
using LinearAlgebra

const OUT = joinpath(@__DIR__, "data")
const N_FOLDS = 5

ols() = LinearRegressionLearner()
logit() = LogisticRegressionLearner(α=1e-6, max_iter=200)

function load_smpls(path)
    raw = JSON.parsefile(path)
    folds = [(train=Int.(f["train"]) .+ 1, test=Int.(f["test"]) .+ 1) for f in raw]
    return [folds]
end

function pack(m, seconds; theta_true=nothing, extra=Dict())
    d = Dict{String,Any}(
        "coef" => collect(Float64, m.coef),
        "se" => collect(Float64, m.se),
        "seconds" => seconds,
        "n_coef" => length(m.coef),
    )
    theta_true !== nothing && (d["theta_true"] = theta_true)
    merge!(d, extra)
    return d
end

function timed(f)
    t0 = time()
    m = f()
    return m, time() - t0
end

function main()
    results = Dict{String,Any}(
        "backend" => "julia",
        "doubleml" => "1.4.0",
        "seed" => 3141,
        "n_folds" => N_FOLDS,
        "models" => Dict{String,Any}(),
    )
    models = results["models"]

    # PLR
    df = CSV.read(joinpath(OUT, "bench_plr.csv"), DataFrame)
    data = DoubleMLData(df; y_col="y", d_cols="d")
    smpls = load_smpls(joinpath(OUT, "bench_plr_smpls.json"))
    m, sec = timed() do
        plr = DoubleMLPLR(data, ols(), ols(); n_folds=N_FOLDS, n_rep=1, draw_sample_splitting=false)
        set_sample_splitting!(plr, smpls)
        fit!(plr)
        plr
    end
    models["PLR"] = pack(m, sec; theta_true=0.5)
    f = construct_framework(m)
    f2 = 2 * f
    models["Framework_2x_PLR"] = Dict(
        "coef" => collect(f2.thetas),
        "se" => collect(f2.ses),
        "seconds" => 0.0,
        "theta_true" => 1.0,
    )
    sensitivity_analysis!(m; cf_y=0.04, cf_d=0.03, rho=1.0, level=0.95, null_hypothesis=0.0)
    r = m.sensitivity
    models["PLR_sensitivity"] = Dict(
        "coef" => collect(m.coef),
        "se" => collect(m.se),
        "theta_lower" => collect(r.theta_lower),
        "theta_upper" => collect(r.theta_upper),
        "ci_lower" => collect(r.ci_lower),
        "ci_upper" => collect(r.ci_upper),
        "rv" => collect(r.rv),
        "rva" => collect(r.rva),
        "seconds" => 0.0,
        "cf_y" => 0.04,
        "cf_d" => 0.03,
        "rho" => 1.0,
    )

    # IRM
    df = CSV.read(joinpath(OUT, "bench_irm.csv"), DataFrame)
    data = DoubleMLData(df; y_col="y", d_cols="d")
    smpls = load_smpls(joinpath(OUT, "bench_irm_smpls.json"))
    m, sec = timed() do
        irm = DoubleMLIRM(data, ols(), logit(); n_folds=N_FOLDS, n_rep=1,
                          draw_sample_splitting=false, trimming_threshold=0.01)
        set_sample_splitting!(irm, smpls)
        fit!(irm)
        irm
    end
    models["IRM"] = pack(m, sec; theta_true=0.5)

    # PLIV
    df = CSV.read(joinpath(OUT, "bench_pliv.csv"), DataFrame)
    zcols = [c for c in names(df) if startswith(String(c), "Z")]
    data = DoubleMLData(df; y_col="y", d_cols="d", z_cols=zcols)
    smpls = load_smpls(joinpath(OUT, "bench_pliv_smpls.json"))
    m, sec = timed() do
        ml = ols()
        pliv = DoubleMLPLIV(data, clone(ml), clone(ml), clone(ml);
                            n_folds=N_FOLDS, n_rep=1, draw_sample_splitting=false)
        set_sample_splitting!(pliv, smpls)
        fit!(pliv)
        pliv
    end
    models["PLIV"] = pack(m, sec; theta_true=1.0)

    # IIVM
    df = CSV.read(joinpath(OUT, "bench_iivm.csv"), DataFrame)
    zcols = [c for c in names(df) if startswith(lowercase(String(c)), "z")]
    data = DoubleMLData(df; y_col="y", d_cols="d", z_cols=zcols)
    smpls = load_smpls(joinpath(OUT, "bench_iivm_smpls.json"))
    m, sec = timed() do
        iivm = DoubleMLIIVM(data, ols(), logit(), logit(); n_folds=N_FOLDS, n_rep=1,
                            draw_sample_splitting=false, trimming_threshold=0.05)
        set_sample_splitting!(iivm, smpls)
        fit!(iivm)
        iivm
    end
    models["IIVM"] = pack(m, sec; theta_true=0.5)

    # multi PLR
    df = CSV.read(joinpath(OUT, "bench_plr_multi.csv"), DataFrame)
    data = DoubleMLData(df; y_col="y", d_cols=["d1", "d2"])
    smpls = load_smpls(joinpath(OUT, "bench_plr_multi_smpls.json"))
    m, sec = timed() do
        plr = DoubleMLPLR(data, ols(), ols(); n_folds=N_FOLDS, n_rep=1, draw_sample_splitting=false)
        set_sample_splitting!(plr, smpls)
        fit!(plr)
        plr
    end
    models["PLR_multi"] = pack(m, sec; theta_true=[0.5, -0.3])

    # PLPR
    df = CSV.read(joinpath(OUT, "bench_plpr.csv"), DataFrame)
    # columns: id, time, y, d, x1...
    xcols = [c for c in names(df) if startswith(lowercase(String(c)), "x")]
    data = DoubleMLData(df; y_col="y", d_cols="d", x_cols=xcols, id_col="id", t_col="time")
    for ap in ("fd_exact", "wg_approx", "cre_general", "cre_normal")
        m, sec = timed() do
            plpr = DoubleMLPLPR(data, ols(), ols(); approach=ap, n_folds=3, n_rep=1,
                                rng=MersenneTwister(3141 + 5))
            fit!(plpr)
            plpr
        end
        models["PLPR_$ap"] = pack(m, sec; theta_true=0.5, extra=Dict("n_folds" => 3))
    end

    # DID two-period
    try
        df = CSV.read(joinpath(OUT, "bench_did.csv"), DataFrame)
        if "y" ∉ names(df) && "y0" in names(df) && "y1" in names(df)
            df.y = df.y1 .- df.y0
        end
        data = DoubleMLData(df; y_col="y", d_cols="d")
        smpls = load_smpls(joinpath(OUT, "bench_did_smpls.json"))
        m, sec = timed() do
            did = DoubleMLDID(data, ols(), logit(); n_folds=N_FOLDS, n_rep=1,
                              draw_sample_splitting=false, score="observational")
            set_sample_splitting!(did, smpls)
            fit!(did)
            did
        end
        models["DID"] = pack(m, sec)
    catch e
        models["DID"] = Dict("error" => string(e), "seconds" => nothing)
    end

    # DID multi — use Julia DGP if Python panel columns mismatch
    try
        if isfile(joinpath(OUT, "bench_did_multi.csv"))
            df = CSV.read(joinpath(OUT, "bench_did_multi.csv"), DataFrame)
            # try common schemas
            idc = "id" in names(df) ? "id" : ("unit" in names(df) ? "unit" : nothing)
            tc = "t" in names(df) ? "t" : ("time" in names(df) ? "time" : nothing)
            if idc !== nothing && tc !== nothing && "y" in names(df) && "d" in names(df)
                data = DoubleMLData(df; y_col="y", d_cols="d", id_col=idc, t_col=tc)
            else
                error("unrecognized did multi columns: $(names(df))")
            end
        else
            error("missing bench_did_multi.csv")
        end
        m, sec = timed() do
            multi = DoubleMLDIDMulti(data, ols(), logit(); n_folds=3,
                                     control_group="never_treated",
                                     gt_combinations=:standard,
                                     rng=MersenneTwister(3141 + 7))
            fit!(multi)
            multi
        end
        models["DID_multi"] = pack(m, sec; extra=Dict("n_att" => length(m.coef)))
    catch e
        # fallback synthetic comparable DGP
        data = make_did_panel_data(n_id=200, n_t=4, dim_x=3, theta=2.0; seed=3141 + 7)
        m, sec = timed() do
            multi = DoubleMLDIDMulti(data, ols(), logit(); n_folds=3, rng=MersenneTwister(3141 + 7))
            fit!(multi)
            multi
        end
        models["DID_multi"] = pack(m, sec; extra=Dict("n_att" => length(m.coef),
                                                      "note" => "julia fallback DGP: $(e)"))
    end

    # RDFlex
    try
        df = CSV.read(joinpath(OUT, "bench_rdd.csv"), DataFrame)
        data = DoubleMLData(df; y_col="y", d_cols="d", score_col="score")
        m, sec = timed() do
            rdd = RDFlex(data, ols(); cutoff=0.0, fuzzy=false, n_folds=3, n_rep=1,
                         n_iterations=2, fs_specification="cutoff",
                         rng=MersenneTwister(3141 + 8))
            fit!(rdd)
            rdd
        end
        models["RDFlex"] = pack(m, sec; extra=Dict("n_iterations" => 2, "h_used" => m.h_used))
    catch e
        models["RDFlex"] = Dict("error" => string(e), "seconds" => nothing)
    end

    # SSM MAR
    try
        df = CSV.read(joinpath(OUT, "bench_ssm_mar.csv"), DataFrame)
        data = DoubleMLData(df; y_col="y", d_cols="d", s_col="s")
        smpls = load_smpls(joinpath(OUT, "bench_ssm_mar_smpls.json"))
        m, sec = timed() do
            ssm = DoubleMLSSM(data, ols(), logit(), logit(); n_folds=N_FOLDS, n_rep=1,
                              draw_sample_splitting=false, score="missing-at-random",
                              trimming_threshold=0.05)
            set_sample_splitting!(ssm, smpls)
            fit!(ssm)
            ssm
        end
        models["SSM_MAR"] = pack(m, sec; theta_true=1.0)
    catch e
        models["SSM_MAR"] = Dict("error" => string(e), "seconds" => nothing)
    end

    # SSM nonignorable
    try
        df = CSV.read(joinpath(OUT, "bench_ssm_ni.csv"), DataFrame)
        zcols = [c for c in names(df) if startswith(lowercase(String(c)), "z")]
        data = DoubleMLData(df; y_col="y", d_cols="d", s_col="s", z_cols=zcols)
        smpls = load_smpls(joinpath(OUT, "bench_ssm_ni_smpls.json"))
        m, sec = timed() do
            ssm = DoubleMLSSM(data, ols(), logit(), logit(); n_folds=N_FOLDS, n_rep=1,
                              draw_sample_splitting=false, score="nonignorable",
                              trimming_threshold=0.05, rng=MersenneTwister(3141 + 10))
            set_sample_splitting!(ssm, smpls)
            fit!(ssm)
            ssm
        end
        models["SSM_nonignorable"] = pack(m, sec; theta_true=1.0)
    catch e
        models["SSM_nonignorable"] = Dict("error" => string(e), "seconds" => nothing)
    end

    out = joinpath(OUT, "benchmark_jl.json")
    open(out, "w") do io
        JSON.print(io, results, 2)
    end
    println(JSON.json(results, 2))
    println("wrote ", out)
end

main()
