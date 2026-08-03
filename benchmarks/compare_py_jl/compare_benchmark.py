#!/usr/bin/env python3
"""Compare Python vs Julia benchmark JSON; write BENCHMARK_REPORT.md"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data"
REP = Path(__file__).resolve().parent / "BENCHMARK_REPORT.md"


def load(path):
    return json.loads(Path(path).read_text())


def fmt(x, nd=6):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def main():
    py = load(OUT / "benchmark_py.json")
    jl = load(OUT / "benchmark_jl.json")
    py_m = py["models"]
    jl_m = jl["models"]

    keys = sorted(set(py_m) | set(jl_m), key=lambda k: (
        0 if k.startswith("PLR") and "multi" not in k and "Framework" not in k else
        1 if k == "IRM" else
        2 if k == "PLIV" else
        3 if k == "IIVM" else
        4 if "multi" in k else
        5 if k.startswith("PLPR") else
        6 if k == "DID" else
        7 if "DID_multi" in k else
        8 if "RDFlex" in k else
        9 if "Framework" in k else 10, k
    ))

    rows = []
    lines = []
    lines.append("# Algorithm Benchmark: Python DoubleML vs Julia DoubleML.jl\n")
    lines.append(f"- **Python**: DoubleML `{py.get('doubleml')}`\n")
    lines.append(f"- **Julia**: DoubleML.jl `{jl.get('doubleml')}`\n")
    lines.append(f"- **Protocol**: shared CSV + shared K-fold (where applicable); OLS / near-unregularized logistic; `n_rep=1`\n")
    lines.append(f"- **Seed**: {py.get('seed')}, default `n_folds={py.get('n_folds')}`\n\n")

    lines.append("## Coefficient & SE comparison\n\n")
    lines.append("| Model | j | Py coef | Jl coef | |Δcoef| | Py SE | Jl SE | |ΔSE| | rel Δcoef | Py s | Jl s | speedup |\n")
    lines.append("|-------|---|--------:|--------:|--------:|------:|------:|------:|----------:|-----:|-----:|--------:|\n")

    for key in keys:
        p = py_m.get(key, {})
        j = jl_m.get(key, {})
        if "error" in p and "error" in j:
            lines.append(f"| {key} | — | error | error | — | — | — | — | — | — | — | — |\n")
            continue
        if "error" in p:
            lines.append(f"| {key} | — | err | {fmt(j.get('coef',[None])[0] if j.get('coef') else None)} | — | — | — | — | — | — | {fmt(j.get('seconds'),3)} | — |\n")
            continue
        if "error" in j:
            lines.append(f"| {key} | — | {fmt(p.get('coef',[None])[0])} | err | — | — | — | — | — | {fmt(p.get('seconds'),3)} | — | — |\n")
            continue

        pc = p.get("coef") or []
        jc = j.get("coef") or []
        ps = p.get("se") or []
        js = j.get("se") or []
        n = max(len(pc), len(jc))
        py_t = p.get("seconds")
        jl_t = j.get("seconds")
        speed = (py_t / jl_t) if (py_t and jl_t and jl_t > 0) else None

        for i in range(n):
            pci = pc[i] if i < len(pc) else None
            jci = jc[i] if i < len(jc) else None
            psi = ps[i] if i < len(ps) else None
            jsi = js[i] if i < len(js) else None
            dcoef = abs(pci - jci) if pci is not None and jci is not None else None
            dse = abs(psi - jsi) if psi is not None and jsi is not None else None
            rel = (dcoef / max(abs(pci), 1e-12)) if dcoef is not None else None
            name = key if i == 0 else ""
            lines.append(
                f"| {name} | {i} | {fmt(pci)} | {fmt(jci)} | {fmt(dcoef,2)} | "
                f"{fmt(psi)} | {fmt(jsi)} | {fmt(dse,2)} | {fmt(rel,2)} | "
                f"{fmt(py_t if i==0 else None, 3)} | {fmt(jl_t if i==0 else None, 3)} | "
                f"{fmt(speed if i==0 else None, 2)} |\n"
            )
            rows.append((key, i, dcoef, rel, py_t, jl_t))

    lines.append("\n## Bias vs true parameter (where known)\n\n")
    lines.append("| Model | j | true | Py bias | Jl bias |\n")
    lines.append("|-------|---|-----:|--------:|--------:|\n")
    for key in keys:
        p = py_m.get(key, {})
        j = jl_m.get(key, {})
        if "theta_true" not in p and "theta_true" not in j:
            continue
        truth = p.get("theta_true", j.get("theta_true"))
        if not isinstance(truth, list):
            truth = [truth]
        pc = p.get("coef") or []
        jc = j.get("coef") or []
        for i, th in enumerate(truth):
            if th is None:
                continue
            pb = (pc[i] - th) if i < len(pc) else None
            jb = (jc[i] - th) if i < len(jc) else None
            lines.append(f"| {key if i==0 else ''} | {i} | {fmt(th)} | {fmt(pb)} | {fmt(jb)} |\n")

    # summary stats
    rels = [r[3] for r in rows if r[3] is not None]
    dcoefs = [r[2] for r in rows if r[2] is not None]
    lines.append("\n## Summary\n\n")
    if rels:
        lines.append(f"- Median relative |Δcoef|: **{sorted(rels)[len(rels)//2]:.2e}**\n")
        lines.append(f"- Max relative |Δcoef|: **{max(rels):.2e}**\n")
        lines.append(f"- Median |Δcoef|: **{sorted(dcoefs)[len(dcoefs)//2]:.2e}**\n")
    # timing totals
    py_tot = sum(py_m[k].get("seconds") or 0 for k in py_m if isinstance(py_m[k].get("seconds"), (int, float)))
    jl_tot = sum(jl_m[k].get("seconds") or 0 for k in jl_m if isinstance(jl_m[k].get("seconds"), (int, float)))
    lines.append(f"- Total timed fit seconds — Python: **{py_tot:.3f}s**, Julia: **{jl_tot:.3f}s**\n")
    if jl_tot > 0:
        lines.append(f"- Overall speedup (Py/Jl total): **{py_tot/jl_tot:.2f}×**\n")

    # sensitivity special section
    if "PLR_sensitivity" in py_m and "PLR_sensitivity" in jl_m:
        p = py_m["PLR_sensitivity"]
        j = jl_m["PLR_sensitivity"]
        lines.append("\n## Sensitivity bounds (PLR, cf_y=0.04, cf_d=0.03)\n\n")
        lines.append("| quantity | Python | Julia | |Δ| |\n")
        lines.append("|----------|-------:|------:|----:|\n")
        for k in ("theta_lower", "theta_upper", "ci_lower", "ci_upper", "rv", "rva"):
            if k in p and k in j:
                pv = float(np.asarray(p[k]).reshape(-1)[0])
                jv = float(np.asarray(j[k]).reshape(-1)[0])
                lines.append(f"| {k} | {pv:.6g} | {jv:.6g} | {abs(pv-jv):.2e} |\n")

    lines.append("\n## Notes\n\n")
    lines.append("- **Linear models (PLR/PLIV/multi/PLPR/Framework)**: expect near bit-level agreement with shared folds + OLS.\n")
    lines.append("- **IRM/IIVM/DID/SSM**: small gaps from logistic solver / nested CF RNG differences.\n")
    lines.append("- **SSM nonignorable**: nested half-splits use stratified shuffle; Julia RNG ≠ sklearn `random_state=42` unless aligned — expect larger gaps than MAR.\n")
    lines.append("- **RDFlex**: Python uses `rdrobust` final stage; Julia uses weighted local linear + residual ROT — coef may differ more than linear DML.\n")
    lines.append("- **DID multi**: never-treated coding (0 vs +inf) and CS internals may differ per-cell ATTs.\n")
    lines.append("- Timings are single-run wall clock; Julia first fits include JIT compile cost.\n")

    lines.append("\n## Reproduce\n\n```bash\n")
    lines.append("python3 benchmarks/compare_py_jl/run_benchmark_python.py\n")
    lines.append("julia --project=. benchmarks/compare_py_jl/run_benchmark_julia.jl\n")
    lines.append("python3 benchmarks/compare_py_jl/compare_benchmark.py\n")
    lines.append("```\n")

    REP.write_text("".join(lines))
    print("".join(lines))
    print("wrote", REP)


if __name__ == "__main__":
    main()
