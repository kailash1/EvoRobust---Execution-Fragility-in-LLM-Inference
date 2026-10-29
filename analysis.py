from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon

from experiment import EXECUTIONS, read_jsonl, write_json
from search import execution_failure_signature, run_method


CONDITIONS = [
    "qwen0p6_bf16_discovery",
    "qwen0p6_bf16_heldout",
    "qwen0p6_bf16_new_prompts",
    "qwen1p7_bf16_discovery",
    "qwen0p6_fp16_discovery",
    "smollm2_1p7b_bf16",
]

REFERENCE_SEARCH_FILES = {
    "qwen0p6_bf16_discovery": "qwen0p6_bf16_discovery_final_metrics.csv",
    "qwen0p6_bf16_heldout": "qwen0p6_bf16_heldout_final_metrics.csv",
    "qwen0p6_bf16_new_prompts": "qwen0p6_bf16_new_prompts_final_metrics.csv",
    "qwen1p7_bf16_discovery": "qwen1p7_bf16_discovery_final_metrics.csv",
    "qwen0p6_fp16_discovery": "qwen0p6_fp16_discovery_final_metrics.csv",
    "smollm2_1p7b_bf16": "cross_family_final_metrics.csv",
}

METHOD_NAMES = {
    "corrected_margin_state_major": "margin_state_major",
    "margin_state_major": "margin_state_major",
    "margin_fail_expand": "margin_fail_expand",
    "bc_qd": "bc_qd",
    "boundary_uniform": "boundary_uniform",
    "boundary_map_elites": "boundary_map_elites",
}


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def auc_binary(y: Sequence[int], score: Sequence[float]) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    u = ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2
    return float(u / (pos.sum() * neg.sum()))


def rule_of_three(n: int) -> float:
    return float(3.0 / n) if n > 0 else float("nan")


def parent_bootstrap(rows: Sequence[dict[str, Any]], threshold: float = 0.25, reps: int = 4000, seed: int = 88000) -> tuple[float, float, float]:
    parents = sorted({r["parent_id"] for r in rows})
    index = {p: i for i, p in enumerate(parents)}
    counts = np.zeros((len(parents), 4), dtype=np.int64)
    for r in rows:
        margin = float(r["ref_margin"])
        if margin <= 0:
            continue
        i = index[r["parent_id"]]
        if margin <= threshold:
            counts[i, 0] += int(bool(r["strict_flip"]))
            counts[i, 1] += 1
        else:
            counts[i, 2] += int(bool(r["strict_flip"]))
            counts[i, 3] += 1

    def calc(c: np.ndarray) -> float:
        x = c.sum(axis=0)
        return float(x[0] / max(1, x[1]) - x[2] / max(1, x[3]))

    observed = calc(counts)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(parents), size=(reps, len(parents)))
    sampled = counts[draw].sum(axis=1)
    values = sampled[:, 0] / np.maximum(1, sampled[:, 1]) - sampled[:, 2] / np.maximum(1, sampled[:, 3])
    lo, hi = np.quantile(values, [0.025, 0.975])
    return observed, float(lo), float(hi)


def diagnostic_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in rows if float(r["ref_margin"]) > 0]
    y = [int(bool(r["strict_flip"])) for r in rows]
    margin = [-float(r["ref_margin"]) for r in rows]
    eta = [float(r["max_competitor_relative_shift"]) for r in rows]
    ratio = [float(r["eta_to_margin_ratio"]) for r in rows]
    indicator = [1.0 if bool(r.get("eta_condition", float(r["max_competitor_relative_shift"]) + 1e-12 >= float(r["ref_margin"]))) else 0.0 for r in rows]
    positives = sum(y)
    predicted = sum(int(v > 0) for v in indicator)
    true_predicted = sum(int(a and b) for a, b in zip(y, indicator))
    return {
        "pairs": len(rows),
        "flips": positives,
        "auc_negative_margin": auc_binary(y, margin),
        "auc_eta": auc_binary(y, eta),
        "auc_eta_over_margin": auc_binary(y, ratio),
        "auc_eta_indicator": auc_binary(y, indicator),
        "eta_indicator_recall": true_predicted / positives if positives else None,
        "eta_indicator_precision": true_predicted / predicted if predicted else None,
    }


def landscape_summary(rows: Sequence[dict[str, Any]], threshold: float = 0.25, reps: int = 4000, seed: int = 0) -> dict[str, Any]:
    positive = [r for r in rows if float(r["ref_margin"]) > 0]
    inside = [r for r in positive if float(r["ref_margin"]) <= threshold]
    above = [r for r in positive if float(r["ref_margin"]) > threshold]
    flips = [r for r in positive if bool(r["strict_flip"])]
    risk, lo, hi = parent_bootstrap(rows, threshold, reps, 88000 + seed)
    return {
        "pairs": len(rows),
        "states": len({r["state_id"] for r in rows}),
        "parents": len({r["parent_id"] for r in rows}),
        "positive_margin_pairs": len(positive),
        "screen_pairs": len(inside),
        "screen_fraction_all_pairs": len(inside) / len(rows),
        "excluded_fraction_all_pairs": 1.0 - len(inside) / len(rows),
        "flips": len(flips),
        "fragile_states": len({r["state_id"] for r in flips}),
        "fragile_parents": len({r["parent_id"] for r in flips}),
        "in_screen_flip_rate": len(flips) / max(1, len(inside)),
        "flips_above_screen": sum(int(bool(r["strict_flip"])) for r in above),
        "zero_count_denominator_above_screen": len(above),
        "zero_count_rule_of_three_95_upper": rule_of_three(len(above)),
        "parent_bootstrap_risk_diff": risk,
        "parent_bootstrap_ci95": [lo, hi],
    }


def threshold_table(rows_by_name: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for name, rows in rows_by_name.items():
        flips = [r for r in rows if bool(r["strict_flip"])]
        for threshold in [0.05, 0.10, 0.125, 0.25, 0.50]:
            out.append({
                "condition": name,
                "threshold": threshold,
                "flips_total": len(flips),
                "flips_at_or_below": sum(int(float(r["ref_margin"]) <= threshold) for r in flips),
                "pairs_at_or_below": sum(int(0 < float(r["ref_margin"]) <= threshold) for r in rows),
                "fraction_full_pool": sum(int(0 < float(r["ref_margin"]) <= threshold) for r in rows) / len(rows),
            })
    return out


def _diagnostic_rows(root: Path, name: str, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    direct = root / "diagnostics" / f"{name}.jsonl"
    if direct.exists():
        return [r for r in read_jsonl(direct) if float(r["ref_margin"]) > 0]
    selected = manifest["diagnostic_states"].get(name)
    if selected is None:
        return [r for r in rows if float(r["ref_margin"]) > 0]
    keep = set(selected)
    return [r for r in rows if r["state_id"] in keep and float(r["ref_margin"]) > 0]


def _tie_counts(name: str, rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> tuple[int, int]:
    source = diagnostics if name != "smollm2_1p7b_bf16" else rows
    flips = [r for r in source if bool(r["strict_flip"])]
    tie = sum(int(bool(r.get("alternative_tie", float(r.get("alt_winner_advantage", 1.0)) == 0.0))) for r in flips)
    strict = sum(int(bool(r.get("strict_superiority", float(r.get("alt_winner_advantage", 0.0)) > 0.0))) for r in flips)
    return tie, strict


def validate_results(root: str | Path, manifest_path: str | Path, check_hashes: bool = False) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    failures: list[str] = []
    details: dict[str, Any] = {}
    total_pairs = total_flips = total_ties = total_strict = 0

    if check_hashes:
        for rel, expected_hash in manifest.get("reference_file_sha256", {}).items():
            if not rel.startswith("reference_results/"):
                continue
            p = Path(manifest_path).parent / rel
            if not p.exists():
                failures.append(f"missing reference file: {rel}")
                continue
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if got != expected_hash:
                failures.append(f"checksum mismatch: {rel}")

    for name in CONDITIONS:
        rows = read_jsonl(root / "matrices" / f"{name}.jsonl")
        diagnostics = _diagnostic_rows(root, name, rows, manifest)
        summary = landscape_summary(rows, 0.25, 4000, CONDITIONS.index(name))
        diag = diagnostic_summary(diagnostics)
        tie, strict = _tie_counts(name, rows, diagnostics)
        expected = manifest["conditions"][name]
        checks = {
            "pairs": len(rows),
            "positive_margin_pairs": summary["positive_margin_pairs"],
            "screen_pairs": summary["screen_pairs"],
            "above_screen_pairs": summary["zero_count_denominator_above_screen"],
            "flips": summary["flips"],
            "tie_flips": tie,
            "strict_superiority_flips": strict,
            "fragile_states": summary["fragile_states"],
            "fragile_parents": summary["fragile_parents"],
            "diag_n": diag["pairs"],
        }
        for key, value in checks.items():
            if value != expected[key]:
                failures.append(f"{name}: {key}={value}, expected {expected[key]}")
        for key in ["auc_negative_margin", "auc_eta", "auc_eta_over_margin", "auc_eta_indicator"]:
            if abs(float(diag[key]) - float(expected[key])) > 5e-10:
                failures.append(f"{name}: {key}={diag[key]:.12f}, expected {expected[key]:.12f}")
        if abs(float(diag["eta_indicator_precision"]) - float(expected["eta_precision"])) > 5e-10:
            failures.append(f"{name}: eta_indicator_precision mismatch")
        if summary["flips_above_screen"] != 0:
            failures.append(f"{name}: observed flip above frozen screen")
        details[name] = {"landscape": summary, "diagnostic": diag, "tie_flips": tie, "strict_superiority_flips": strict}
        total_pairs += len(rows)
        total_flips += summary["flips"]
        total_ties += tie
        total_strict += strict

    totals = {"pairs": total_pairs, "flips": total_flips, "tie_flips": total_ties, "strict_superiority_flips": total_strict}
    for key, value in manifest["expected_totals"].items():
        if totals[key] != value:
            failures.append(f"total {key}={totals[key]}, expected {value}")

    return {"pass": not failures, "failures": failures, "totals": totals, "conditions": details}


def validate_reference_search(root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    failures = []
    observed = {}
    for name, filename in REFERENCE_SEARCH_FILES.items():
        with (root / "search" / filename).open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        reachable = manifest["conditions"][name]["reachable_signatures"]
        by_method: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            method = METHOD_NAMES.get(row["method"])
            if method:
                by_method[method].append(float(row["execution_failure_signatures_auc"]))
        observed[name] = {}
        for method, expected in manifest["expected_search_normalized_auc_percent"][name].items():
            values = by_method.get(method, [])
            if not values:
                failures.append(f"{name}: missing search method {method}")
                continue
            value = 100.0 * statistics.mean(values) / reachable
            observed[name][method] = value
            if abs(value - float(expected)) > 0.11:
                failures.append(f"{name}/{method}: normalized AUC {value:.3f}, expected paper {expected:.1f}")
    return {"pass": not failures, "failures": failures, "observed": observed}


def search_analysis(rows_by_name: dict[str, list[dict[str, Any]]], out_dir: Path, n_seeds: int = 20) -> dict[str, Any]:
    canonical = [e.name for e in EXECUTIONS]
    key_methods = ["margin_state_major", "margin_fail_expand", "bc_qd", "boundary_uniform", "boundary_map_elites"]
    ablations = ["margin_breadth", "boundary_only", "bc_qd_no_boundary", "bc_qd_no_coverage", "bc_qd_no_novelty", "bc_qd_cumulative_credit"]
    all_metrics = {}
    all_checkpoints = {}
    normalized = []
    paired = {}
    checkpoints_csv = []
    budgets = [25, 50, 100, 250, 500]

    for name, rows in rows_by_name.items():
        reachable = len({execution_failure_signature(r) for r in rows if bool(r["strict_flip"]) and 0 < float(r["ref_margin"]) <= 0.25})
        methods = list(key_methods)
        if name in ("qwen0p6_bf16_discovery", "qwen0p6_bf16_new_prompts"):
            methods += ablations
        metrics = []
        checkpoints = []
        for method in methods:
            seeds = [940000] if method in ("margin_state_major", "margin_breadth") else range(940000, 940000 + n_seeds)
            for seed in seeds:
                history, met = run_method(rows, method, seed, 500, 0.25, canonical)
                metrics.append(met)
                normalized.append({
                    "condition": name,
                    "method": method,
                    "seed": seed,
                    "reachable_signatures": reachable,
                    "normalized_signature_auc_percent": 100.0 * met["execution_failure_signatures_auc"] / reachable if reachable else float("nan"),
                    "raw_signature_auc": met["execution_failure_signatures_auc"],
                })
                for budget in budgets:
                    h = history[:min(budget, len(history))]
                    flips = [r for r in h if bool(r.get("strict_flip"))]
                    row = {
                        "condition": name,
                        "method": method,
                        "seed": seed,
                        "requested_budget": budget,
                        "budget": len(h),
                        "execution_failure_signatures": len({execution_failure_signature(r) for r in flips}),
                        "fragile_states": len({r["state_id"] for r in flips}),
                        "fragile_parents": len({r["parent_id"] for r in flips}),
                        "strict_failures": len(flips),
                    }
                    checkpoints.append(row)
                    checkpoints_csv.append(row)
        all_metrics[name] = metrics
        all_checkpoints[name] = checkpoints
        bc = {r["seed"]: r["execution_failure_signatures_auc"] for r in metrics if r["method"] == "bc_qd"}
        fx = {r["seed"]: r["execution_failure_signatures_auc"] for r in metrics if r["method"] == "margin_fail_expand"}
        seeds = sorted(set(bc) & set(fx))
        if seeds:
            diff = [bc[s] - fx[s] for s in seeds]
            stat, p = wilcoxon([bc[s] for s in seeds], [fx[s] for s in seeds], zero_method="wilcox")
            paired[name] = {"n": len(seeds), "mean_bc_minus_fail_expand": statistics.mean(diff), "wilcoxon_stat": float(stat), "p": float(p)}

    write_csv(normalized, out_dir / "normalized_search_auc.csv")
    write_csv(checkpoints_csv, out_dir / "full_budget_trajectory.csv")
    write_csv([{"condition": k, **v} for k, v in paired.items()], out_dir / "paired_wilcoxon.csv")
    return {"metrics": all_metrics, "checkpoints": all_checkpoints, "paired": paired}


def strict_superiority_sensitivity(rows_by_name: dict[str, list[dict[str, Any]]], diagnostic_by_name: dict[str, list[dict[str, Any]]], out_dir: Path) -> list[dict[str, Any]]:
    canonical = [e.name for e in EXECUTIONS]
    out = []
    for name in ("qwen0p6_bf16_discovery", "qwen0p6_bf16_new_prompts"):
        rows = rows_by_name[name]
        strict_pairs = {(r["state_id"], r["execution"]) for r in diagnostic_by_name[name] if bool(r.get("strict_superiority"))}
        relabeled = []
        for r in rows:
            x = dict(r)
            x["strict_flip"] = (r["state_id"], r["execution"]) in strict_pairs
            relabeled.append(x)
        for seed in range(940000, 940005):
            for method in ("margin_fail_expand", "bc_qd"):
                _, metrics = run_method(relabeled, method, seed, 500, 0.25, canonical)
                out.append({"condition": name, "method": method, "seed": seed, "strict_superiority_auc": metrics["execution_failure_signatures_auc"]})
    write_csv(out, out_dir / "strict_superiority_search.csv")
    return out


def _nice_ceiling(value: float, step: float) -> float:
    return step * math.ceil(value / step)


def _flip_plot_class(row: dict[str, Any]) -> str:
    """Classify archival or freshly generated rows without silently dropping flips.

    Older enriched diagnostic files store alt_winner_advantage, but lack the
    alternative_tie and strict_superiority flags added by decision_diagnostics.
    """
    if not bool(row.get("strict_flip")):
        return "nonflip"
    if "alt_winner_advantage" in row:
        advantage = float(row["alt_winner_advantage"])
        if advantage < 0:
            raise ValueError("A reported top-1 flip has negative winner advantage")
        label = "tie" if advantage == 0 else "strict"
    elif "alternative_tie" in row and "strict_superiority" in row:
        if bool(row["alternative_tie"]) == bool(row["strict_superiority"]):
            raise ValueError("Flip must be exactly one of tie or strict superiority")
        label = "tie" if bool(row["alternative_tie"]) else "strict"
    else:
        raise ValueError("No tie/superiority classification on a reported flip")
    return label


def make_figure(diagnostic_by_name: dict[str, list[dict[str, Any]]], search_payload: dict[str, Any], rows_by_name: dict[str, list[dict[str, Any]]], out_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    pooled = [r for rows in diagnostic_by_name.values() for r in rows if float(r["ref_margin"]) > 0]
    xmax = _nice_ceiling(max(float(r["ref_margin"]) for r in pooled), 2.5)
    ymax = _nice_ceiling(max(float(r["max_competitor_relative_shift"]) for r in pooled), 0.5)

    fig = plt.figure(figsize=(10.4, 4.1))
    ax = fig.add_subplot(1, 2, 1)
    classified = [(r, _flip_plot_class(r)) for r in pooled]
    nonflip = [r for r, cls in classified if cls == "nonflip"]
    ties = [r for r, cls in classified if cls == "tie"]
    strict = [r for r, cls in classified if cls == "strict"]
    ax.scatter([r["ref_margin"] for r in nonflip], [r["max_competitor_relative_shift"] for r in nonflip], s=3, alpha=0.035, label="non-flip")
    ax.scatter([r["ref_margin"] for r in ties], [r["max_competitor_relative_shift"] for r in ties], s=13, alpha=0.8, marker="o", label="tie-resolved flip")
    ax.scatter([r["ref_margin"] for r in strict], [r["max_competitor_relative_shift"] for r in strict], s=18, alpha=0.9, marker="x", label="strict-superiority flip")
    diag_max = min(xmax, ymax)
    ax.plot([0, diag_max], [0, diag_max], linestyle="--", linewidth=1, label="eta = m0")
    ax.axvline(0.25, linestyle=":", linewidth=1, label="frozen m0 = .25")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Reference margin m0")
    ax.set_ylabel("Decision-directed shift eta")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    inset = ax.inset_axes([0.38, 0.51, 0.40, 0.34])
    low_nonflips = [r for r in nonflip if 0 < float(r["ref_margin"]) <= 0.30]
    low_ties = [r for r in ties if 0 < float(r["ref_margin"]) <= 0.30]
    low_strict = [r for r in strict if 0 < float(r["ref_margin"]) <= 0.30]
    inset.scatter([r["ref_margin"] for r in low_nonflips], [r["max_competitor_relative_shift"] for r in low_nonflips], s=3, alpha=0.08)
    inset.scatter([r["ref_margin"] for r in low_ties], [r["max_competitor_relative_shift"] for r in low_ties], s=10, alpha=0.8, marker="o")
    inset.scatter([r["ref_margin"] for r in low_strict], [r["max_competitor_relative_shift"] for r in low_strict], s=13, alpha=0.9, marker="x")
    inset.plot([0, 0.3], [0, 0.3], linestyle="--", linewidth=0.8)
    inset.axvline(0.25, linestyle=":", linewidth=0.8)
    inset.set_xlim(0, 0.30)
    inset.set_ylim(0, 1.0)
    inset.set_title("low-margin region", fontsize=7)

    ax2 = fig.add_subplot(1, 2, 2)
    discovery = search_payload["checkpoints"]["qwen0p6_bf16_discovery"]
    reachable = len({execution_failure_signature(r) for r in rows_by_name["qwen0p6_bf16_discovery"] if bool(r["strict_flip"]) and 0 < float(r["ref_margin"]) <= 0.25})
    by_method: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in discovery:
        by_method[row["method"]][row["requested_budget"]].append(row["execution_failure_signatures"])
    for method in ("margin_fail_expand", "bc_qd", "margin_state_major", "boundary_uniform", "boundary_map_elites"):
        if method not in by_method:
            continue
        xs = sorted(by_method[method])
        ys = [statistics.mean(by_method[method][b]) / reachable for b in xs]
        ax2.plot(xs, ys, marker="o", label=method)
    ax2.set_xlabel("Queries inside frozen screen")
    ax2.set_ylabel(f"Fraction of {reachable} reachable signatures")
    ax2.set_xticks([0, 100, 250, 500])
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _load_reference_checkpoints(reference_root: Path) -> dict[str, Any]:
    path = reference_root / "search" / "qwen0p6_bf16_discovery_checkpoint_metrics.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    checkpoints = []
    aliases = {"corrected_margin_state_major": "margin_state_major"}
    for row in rows:
        method = aliases.get(row["method"], row["method"])
        if method not in ("margin_fail_expand", "bc_qd", "margin_state_major", "boundary_uniform", "boundary_map_elites"):
            continue
        checkpoints.append({
            "method": method,
            "seed": int(row["seed"]),
            "requested_budget": int(row["requested_budget"]),
            "execution_failure_signatures": int(float(row["execution_failure_signatures"])),
        })
    return {"checkpoints": {"qwen0p6_bf16_discovery": checkpoints}}


def reproduce_reference_figure(reference_root: str | Path, manifest_path: str | Path, output: str | Path) -> None:
    root = Path(reference_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows_by_name = {name: read_jsonl(root / "matrices" / f"{name}.jsonl") for name in CONDITIONS}
    diagnostics = {name: _diagnostic_rows(root, name, rows_by_name[name], manifest) for name in CONDITIONS}
    search_payload = _load_reference_checkpoints(root)
    make_figure(diagnostics, search_payload, rows_by_name, output)


def analyze(results_dir: str | Path, manifest_path: str | Path, n_seeds: int = 20, validate: bool = True) -> dict[str, Any]:
    root = Path(results_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows_by_name = {name: read_jsonl(root / "matrices" / f"{name}.jsonl") for name in CONDITIONS}
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    landscape = {}
    diagnostics = {}
    diagnostic_by_name = {}
    screen_rows = []
    diagnostic_rows = []
    for i, name in enumerate(CONDITIONS):
        rows = rows_by_name[name]
        landscape[name] = landscape_summary(rows, 0.25, 4000, i)
        drows = _diagnostic_rows(root, name, rows, manifest)
        diagnostic_by_name[name] = drows
        diagnostics[name] = diagnostic_summary(drows)
        screen_rows.append({"condition": name, **landscape[name]})
        diagnostic_rows.append({"condition": name, **diagnostics[name]})

    write_csv(screen_rows, tables / "screen_denominators.csv")
    write_csv(diagnostic_rows, tables / "diagnostic_auc.csv")
    write_csv(threshold_table(rows_by_name), tables / "threshold_sensitivity.csv")
    search_payload = search_analysis(rows_by_name, tables, n_seeds)
    strict_payload = strict_superiority_sensitivity(rows_by_name, diagnostic_by_name, tables)
    make_figure(diagnostic_by_name, search_payload, rows_by_name, figures / "main_geometry_search.png")

    repeat_rows = []
    for fp in sorted(root.glob("workers/*/*/repeatability.json")) + sorted(root.glob("workers/*/repeatability.json")):
        repeat_rows.extend(json.loads(fp.read_text(encoding="utf-8")))
    write_csv(repeat_rows, tables / "repeatability.csv")

    payload = {
        "landscape": landscape,
        "diagnostics": diagnostics,
        "paired_wilcoxon": search_payload["paired"],
        "strict_superiority_sensitivity": strict_payload,
        "repeatability": {
            "checks": len(repeat_rows),
            "top1_disagreements": sum(int(not r["top1_equal"]) for r in repeat_rows),
            "bitwise_disagreements": sum(int(not r["bitwise_equal_logits"]) for r in repeat_rows),
            "max_abs_delta": max([float(r["max_abs_delta"]) for r in repeat_rows] or [0.0]),
        },
    }
    if validate:
        payload["paper_validation"] = validate_results(root, manifest_path, check_hashes=False)
        if not payload["paper_validation"]["pass"]:
            raise RuntimeError("generated results do not match paper manifest: " + "; ".join(payload["paper_validation"]["failures"][:10]))
    write_json(payload, root / "paper_results.json")
    return payload
