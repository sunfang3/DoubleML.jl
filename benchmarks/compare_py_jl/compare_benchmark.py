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


def is_bookkeeping(entry: dict) -> bool:
    if entry.get("exclude_from_runtime_total"):
        return True
    proto = entry.get("seconds_protocol")
    return proto == "bookkeeping"


def timed_seconds(entry: dict):
    """Warm seconds (primary gate metric)."""
    s = entry.get("seconds")
    if isinstance(s, (int, float)) and not is_bookkeeping(entry):
        return float(s)
    return None


def cold_seconds(entry: dict):
    s = entry.get("seconds_cold")
    if isinstance(s, (int, float)) and not is_bookkeeping(entry):
        return float(s)
    return None


def main():
    py = load(OUT / "benchmark_py.json")
    jl = load(OUT / "benchmark_jl.json")
    py_m = py["models"]
    jl_m = jl["models"]

    keys = sorted(set(py_m) | set(jl_m), key=lambda k: (
        0 if k.startswith("PLR") and "multi" not in k and "Framework" not in k and "sensitivity" not in k else
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
    proto = py.get("timing_protocol") or jl.get("timing_protocol") or "single"
    n_warm = py.get("n_warm_reps") or jl.get("n_warm_reps") or 1
    n_cold = py.get("n_cold_discard") or jl.get("n_cold_discard") or 0
    lines.append("# Algorithm Benchmark: Python DoubleML vs Julia DoubleML.jl\n\n")
    lines.append(f"- **Python**: DoubleML `{py.get('doubleml')}`\n")
    lines.append(f"- **Julia**: DoubleML.jl `{jl.get('doubleml')}`\n")
    lines.append(f"- **Protocol**: shared CSV + shared K-fold (where applicable); OLS / near-unregularized logistic; `n_rep=1`\n")
    lines.append(f"- **Timing protocol**: `{proto}` (discard {n_cold} cold fit; median of {n_warm} warm fits). `seconds` = warm median.\n")
    lines.append(f"- **Seed**: {py.get('seed')}, default `n_folds={py.get('n_folds')}`\n")
    lines.append(f"- **Runtime gate**: `(t_jl − t_py) / t_py ≤ 0.30` on **sum of warm** seconds over timed main models (bookkeeping rows excluded).\n\n")

    lines.append("## Coefficient & SE comparison\n\n")
    lines.append("| Model | j | Py coef | Jl coef | |Δcoef| | Py SE | Jl SE | |ΔSE| | rel Δcoef | Py warm s | Jl warm s | speedup |\n")
    lines.append("|-------|---|--------:|--------:|--------:|------:|------:|------:|----------:|----------:|----------:|--------:|\n")

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
        speed = (py_t / jl_t) if (isinstance(py_t, (int, float)) and isinstance(jl_t, (int, float)) and jl_t > 0) else None

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

    # summary stats (exclude DID_multi from max/median — near-zero cells inflate rel)
    main_rows = [r for r in rows if r[0] != "DID_multi"]
    rels = [r[3] for r in main_rows if r[3] is not None]
    dcoefs = [r[2] for r in main_rows if r[2] is not None]
    lines.append("\n## Summary (accuracy)\n\n")
    if rels:
        lines.append(f"- Median relative |Δcoef| (excl. DID multi): **{sorted(rels)[len(rels)//2]:.2e}**\n")
        lines.append(f"- Max relative |Δcoef| (excl. DID multi): **{max(rels):.2e}**\n")
        lines.append(f"- Median |Δcoef| (excl. DID multi): **{sorted(dcoefs)[len(dcoefs)//2]:.2e}**\n")
        within30 = sum(1 for r in rels if r <= 0.30)
        lines.append(f"- Main-model cells within 30% rel Δ: **{within30}/{len(rels)}**\n")
    if "DID_multi" in py_m and "DID_multi" in jl_m:
        pc = np.asarray(py_m["DID_multi"].get("coef") or [], float)
        jc = np.asarray(jl_m["DID_multi"].get("coef") or [], float)
        if len(pc) and len(jc) == len(pc):
            dm_rel = np.abs(pc - jc) / np.maximum(np.abs(pc), 1e-12)
            lines.append(
                f"- DID multi: n_ATT={len(pc)}, median cell rel={np.median(dm_rel):.2e}, "
                f"frac ≤30%={(dm_rel <= 0.30).mean():.0%}\n"
            )

    # Warm / cold timing totals (gating metric = warm sum over timed models)
    timed_keys = sorted(
        k for k in set(py_m) | set(jl_m)
        if timed_seconds(py_m.get(k, {})) is not None or timed_seconds(jl_m.get(k, {})) is not None
    )
    py_warm = sum(timed_seconds(py_m[k]) or 0.0 for k in timed_keys if k in py_m)
    jl_warm = sum(timed_seconds(jl_m[k]) or 0.0 for k in timed_keys if k in jl_m)
    py_cold = sum(cold_seconds(py_m[k]) or 0.0 for k in timed_keys if k in py_m)
    jl_cold = sum(cold_seconds(jl_m[k]) or 0.0 for k in timed_keys if k in jl_m)
    gap = (jl_warm - py_warm) / py_warm if py_warm > 0 else float("inf")
    runtime_pass = bool(py_warm > 0 and gap <= 0.30)

    lines.append("\n## Runtime (warm multi-rep gate)\n\n")
    lines.append("| Aggregate | Python | Julia |\n")
    lines.append("|-----------|-------:|------:|\n")
    lines.append(f"| **Warm total** (gating) | **{py_warm:.4f}s** | **{jl_warm:.4f}s** |\n")
    lines.append(f"| Cold total (discarded) | {py_cold:.4f}s | {jl_cold:.4f}s |\n")
    lines.append(f"| Ratio t_jl / t_py (warm) | — | **{jl_warm/py_warm:.3f}×** |\n" if py_warm > 0 else "| Ratio | — | — |\n")
    lines.append(f"| Relative gap (t_jl − t_py)/t_py | — | **{gap:.2%}** |\n")
    lines.append(f"| Gate (≤ 30%) | — | **{'PASS' if runtime_pass else 'FAIL'}** |\n\n")
    lines.append(f"**RUNTIME_GAP_PASS: {str(runtime_pass).lower()}**\n\n")
    lines.append("Per-model warm seconds:\n\n")
    lines.append("| Model | Py warm | Jl warm | Py cold | Jl cold | Jl/Py |\n")
    lines.append("|-------|--------:|--------:|--------:|--------:|------:|\n")
    for k in timed_keys:
        pw = timed_seconds(py_m.get(k, {}))
        jw = timed_seconds(jl_m.get(k, {}))
        pc_ = cold_seconds(py_m.get(k, {}))
        jc_ = cold_seconds(jl_m.get(k, {}))
        ratio = (jw / pw) if (pw and jw and pw > 0) else None
        lines.append(
            f"| {k} | {fmt(pw, 4)} | {fmt(jw, 4)} | {fmt(pc_, 4)} | {fmt(jc_, 4)} | {fmt(ratio, 2)} |\n"
        )

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
    lines.append("- **Timing**: `seconds` is the **median of warm fits** after discarding cold/JIT. Cold is reported separately and is **not** the 30% gate.\n")
    lines.append("- Framework / sensitivity rows are bookkeeping (re-use fit; excluded from runtime total).\n")

    lines.append("\n## Reproduce\n\n```bash\n")
    lines.append("python3 benchmarks/compare_py_jl/run_benchmark_python.py\n")
    lines.append("julia --project=. benchmarks/compare_py_jl/run_benchmark_julia.jl\n")
    lines.append("python3 benchmarks/compare_py_jl/compare_benchmark.py\n")
    lines.append("```\n")

    text = "".join(lines)
    REP.write_text(text)
    print(text)
    print("wrote", REP)
    print(f"RUNTIME_GAP_PASS: {str(runtime_pass).lower()}  t_py={py_warm:.4f} t_jl={jl_warm:.4f} gap={gap:.2%}")


if __name__ == "__main__":
    main()
