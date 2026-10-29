from __future__ import annotations

import math
import random
import statistics
import time
from collections import Counter, defaultdict, deque
from typing import Any, Sequence


MARGIN_EDGES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0]
CONTEXT_EDGES = [128, 512, 1024, 2048, 8192]
DELTA_EDGES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]


def _bin(v: float, edges: Sequence[float], prefix: str) -> str:
    for x in edges:
        if v <= x:
            return f"{prefix}<={x:g}"
    return f"{prefix}>{edges[-1]:g}"


def margin_bin(v: float) -> str:
    return _bin(float(v), MARGIN_EDGES, "m")


def context_bin(v: int) -> str:
    return _bin(float(v), CONTEXT_EDGES, "L")


def delta_bin(v: float) -> str:
    return _bin(float(v), DELTA_EDGES, "d")


def cell(row: dict[str, Any]) -> str:
    return "|".join((margin_bin(row["ref_margin"]), context_bin(row["prefix_len"]), row["perturbation_family"], row["shape_regime"]))


def failure_signature(row: dict[str, Any]) -> str:
    if not row.get("strict_flip"):
        return ""
    flip = "ref_top2" if row.get("alt_is_ref_top2") else "other_token"
    return "|".join((margin_bin(row["ref_margin"]), context_bin(row["prefix_len"]), row["perturbation_family"], row["shape_regime"], flip, delta_bin(row["max_abs_delta"])))


def execution_failure_signature(row: dict[str, Any]) -> str:
    if not row.get("strict_flip"):
        return ""
    flip = "ref_top2" if row.get("alt_is_ref_top2") else "other_token"
    return "|".join((context_bin(row["prefix_len"]), row["perturbation_family"], row["shape_regime"], flip, delta_bin(row["max_abs_delta"])))


def _auc(curve: Sequence[int | float]) -> float:
    if not curve:
        return 0.0
    prev = 0.0
    area = 0.0
    for y in curve:
        y = float(y)
        area += 0.5 * (prev + y)
        prev = y
    return area / len(curve)


def _metrics(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    flips = [r for r in history if r.get("strict_flip")]
    return {
        "evaluations": len(history),
        "strict_failures": len(flips),
        "fragile_states": len({r["state_id"] for r in flips}),
        "fragile_parents": len({r["parent_id"] for r in flips}),
        "failure_signatures": len({failure_signature(r) for r in flips}),
        "execution_failure_signatures": len({execution_failure_signature(r) for r in flips}),
        "fragile_cells": len({cell(r) for r in flips}),
    }


class Oracle:
    pre_fields = ("state_id", "parent_id", "source", "prefix_len", "execution", "perturbation_family", "shape_regime", "ref_margin")

    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = {(r["state_id"], r["execution"]): r for r in rows}
        self.meta = {k: {f: v[f] for f in self.pre_fields} for k, v in self.rows.items()}
        self.calls = set()

    def evaluate(self, pair: tuple[str, str]) -> dict[str, Any]:
        if pair in self.calls:
            raise RuntimeError(f"repeated query: {pair}")
        self.calls.add(pair)
        return dict(self.rows[pair])


def _state_order(meta: dict[tuple[str, str], dict[str, Any]], eligible: set[tuple[str, str]], rng: random.Random | None = None) -> list[str]:
    by_state = {}
    for p in eligible:
        by_state[p[0]] = meta[p]
    if rng is None:
        return [x["state_id"] for x in sorted(by_state.values(), key=lambda x: (float(x["ref_margin"]), int(x["prefix_len"]), x["state_id"]))]
    groups = defaultdict(list)
    for sid, m in by_state.items():
        groups[float(m["ref_margin"])].append(sid)
    out = []
    for m in sorted(groups):
        xs = sorted(groups[m])
        rng.shuffle(xs)
        out.extend(xs)
    return out


def _exec_order(meta: dict[tuple[str, str], dict[str, Any]], canonical: Sequence[str]) -> list[str]:
    present = {p[1] for p in meta}
    return [x for x in canonical if x in present]


def _state_major(states: list[str], execs: list[str], eligible: set[tuple[str, str]], rng: random.Random | None = None, random_per_state: bool = False) -> list[tuple[str, str]]:
    out = []
    for sid in states:
        eo = list(execs)
        if random_per_state and rng is not None:
            rng.shuffle(eo)
        out.extend((sid, e) for e in eo if (sid, e) in eligible)
    return out


def _breadth(states: list[str], execs: list[str], eligible: set[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(sid, e) for e in execs for sid in states if (sid, e) in eligible]


def _sample_candidates(rng: random.Random, unseen: set[tuple[str, str]], n: int = 32) -> list[tuple[str, str]]:
    xs = sorted(unseen)
    return xs if len(xs) <= n else rng.sample(xs, n)


def _select_emitter(rng: random.Random, emitter: str, unseen: set[tuple[str, str]], meta: dict[tuple[str, str], dict[str, Any]], history: list[dict[str, Any]]) -> tuple[str, str]:
    cand = _sample_candidates(rng, unseen)
    flips = [r for r in history if r.get("strict_flip")]
    occupied = Counter(cell(r) for r in history)
    if flips and emitter in ("novelty", "boundary") and rng.random() < 0.55:
        base = rng.choice(flips)
        local = sorted(p for p in unseen if p[0] == base["state_id"] or meta[p]["parent_id"] == base["parent_id"])
        if local:
            cand = list(dict.fromkeys(rng.sample(local, min(96, len(local))) + cand))

    if emitter == "coverage":
        def score(p):
            m = meta[p]
            c = "|".join((margin_bin(m["ref_margin"]), context_bin(m["prefix_len"]), m["perturbation_family"], m["shape_regime"]))
            src_ctx = sum(1 for r in history if r["source"] == m["source"] and context_bin(r["prefix_len"]) == context_bin(m["prefix_len"]))
            return (-occupied[c], -0.02 * src_ctx, -float(m["ref_margin"]), rng.random())
        return max(cand, key=score)

    if emitter == "novelty":
        refs = flips if flips else history
        seen_fs = {(r["perturbation_family"], r["shape_regime"]) for r in refs}
        seen_sc = {(r["source"], context_bin(r["prefix_len"])) for r in refs}
        seen_mc = {(margin_bin(r["ref_margin"]), context_bin(r["prefix_len"])) for r in refs}
        count_fs = Counter((r["perturbation_family"], r["shape_regime"]) for r in refs)
        count_sc = Counter((r["source"], context_bin(r["prefix_len"])) for r in refs)

        def score(p):
            m = meta[p]
            fs = (m["perturbation_family"], m["shape_regime"])
            sc = (m["source"], context_bin(m["prefix_len"]))
            mc = (margin_bin(m["ref_margin"]), context_bin(m["prefix_len"]))
            val = 3.0 * (fs not in seen_fs) + 2.0 * (sc not in seen_sc) + 1.0 * (mc not in seen_mc) - 0.10 * count_fs[fs] - 0.05 * count_sc[sc]
            return (val, -float(m["ref_margin"]), rng.random())
        return max(cand, key=score)

    by_state = defaultdict(list)
    for r in history:
        by_state[r["state_id"]].append(r)

    def score(p):
        m = meta[p]
        prev = by_state.get(m["state_id"], [])
        prior_flip = any(r.get("strict_flip") for r in prev)
        risk = max([float(r.get("risk_score", 0.0)) for r in prev] or [0.0])
        return (3.0 * prior_flip + min(risk, 8.0), -float(m["ref_margin"]), rng.random())
    return max(cand, key=score)


def _emitters(method: str) -> list[str]:
    if method in ("bc_qd", "bc_qd_static_equal", "bc_qd_cumulative_credit"):
        return ["coverage", "boundary", "novelty"]
    if method == "bc_qd_no_novelty":
        return ["coverage", "boundary"]
    if method == "bc_qd_no_coverage":
        return ["boundary", "novelty"]
    if method == "bc_qd_no_boundary":
        return ["coverage", "novelty"]
    if method == "coverage_only":
        return ["coverage"]
    if method == "novelty_only":
        return ["novelty"]
    if method == "boundary_only":
        return ["boundary"]
    raise ValueError(method)


def _choose_emitter(method: str, names: list[str], credits: dict[str, list[float]], rng: random.Random) -> str:
    if len(names) == 1:
        return names[0]
    if method == "bc_qd_static_equal":
        return rng.choice(names)
    scores = []
    for name in names:
        vals = credits[name] if method == "bc_qd_cumulative_credit" else credits[name][-32:]
        scores.append(sum(vals) / len(vals) if vals else 0.0)
    mx = max(scores)
    weights = [math.exp(4.0 * (x - mx)) for x in scores]
    z = rng.random()
    acc = 0.0
    for name, weight in zip(names, weights):
        acc += weight / sum(weights)
        if z <= acc:
            return name
    return names[-1]


def run_method(rows: Sequence[dict[str, Any]], method: str, seed: int, budget: int, threshold: float, canonical_execs: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oracle = Oracle(rows)
    eligible = {p for p, m in oracle.meta.items() if 0.0 < float(m["ref_margin"]) <= threshold}
    budget = min(int(budget), len(eligible))
    rng = random.Random(seed)
    unseen = set(eligible)
    history = []
    credits = defaultdict(list)
    curves = defaultdict(list)
    execs = _exec_order(oracle.meta, canonical_execs)
    deterministic_states = _state_order(oracle.meta, eligible)
    random_states = _state_order(oracle.meta, eligible, rng)
    random_execs = list(execs)
    rng.shuffle(random_execs)

    schedules = {
        "margin_state_major": _state_major(deterministic_states, execs, eligible),
        "margin_breadth": _breadth(deterministic_states, execs, eligible),
        "margin_random_ties": _state_major(random_states, execs, eligible, rng, True),
    }
    fail_schedule = _breadth(random_states, random_execs, eligible)
    cursor = 0
    priority = deque()
    expanded = set()
    start = time.perf_counter()

    for k in range(budget):
        emitter = ""
        if method in schedules:
            schedule = schedules[method]
            while cursor < len(schedule) and schedule[cursor] not in unseen:
                cursor += 1
            pair = schedule[cursor]
            cursor += 1
        elif method == "margin_fail_expand":
            while priority and priority[0] not in unseen:
                priority.popleft()
            if priority:
                pair = priority.popleft()
            else:
                while cursor < len(fail_schedule) and fail_schedule[cursor] not in unseen:
                    cursor += 1
                pair = fail_schedule[cursor]
                cursor += 1
        elif method == "boundary_uniform":
            pair = rng.choice(sorted(unseen))
        elif method == "boundary_map_elites":
            if k < 12 or not history:
                pair = rng.choice(sorted(unseen))
            else:
                base = rng.choice(history)
                local = sorted(p for p in unseen if p[0] == base["state_id"] or oracle.meta[p]["parent_id"] == base["parent_id"])
                if rng.random() < 0.75:
                    same_family = [p for p in local if oracle.meta[p]["perturbation_family"] == base["perturbation_family"]]
                    if same_family:
                        local = same_family
                pair = rng.choice(local) if local else rng.choice(sorted(unseen))
        else:
            if k < 12 or not history:
                pair = rng.choice(sorted(unseen))
                emitter = "warmup"
            else:
                names = _emitters(method)
                emitter = _choose_emitter(method, names, credits, rng)
                pair = _select_emitter(rng, emitter, unseen, oracle.meta, history)

        unseen.remove(pair)
        before_exec = {execution_failure_signature(r) for r in history if r.get("strict_flip")}
        before_fail = {failure_signature(r) for r in history if r.get("strict_flip")}
        before_cells = {cell(r) for r in history}
        row = oracle.evaluate(pair)
        history.append(row)

        if method == "margin_fail_expand" and row.get("strict_flip") and row["state_id"] not in expanded:
            expanded.add(row["state_id"])
            rem = [(row["state_id"], e) for e in random_execs if (row["state_id"], e) in unseen]
            for p in reversed(rem):
                priority.appendleft(p)

        if method.startswith("bc_qd") and emitter not in ("", "warmup"):
            reward = 0.0
            if row.get("strict_flip"):
                es = execution_failure_signature(row)
                fs = failure_signature(row)
                reward = 1.0 if es not in before_exec else 0.5 if fs not in before_fail else 0.1
            elif cell(row) not in before_cells:
                reward = 0.02
            credits[emitter].append(reward)

        m = _metrics(history)
        for key in ("strict_failures", "fragile_states", "fragile_parents", "failure_signatures", "execution_failure_signatures", "fragile_cells"):
            curves[key].append(m[key])

    final = _metrics(history)
    for key, curve in curves.items():
        final[f"{key}_auc"] = _auc(curve)
    final.update({"method": method, "seed": seed, "budget": len(history), "eligible_pairs": len(eligible), "cpu_seconds": time.perf_counter() - start})
    return history, final


def run_suite(rows: Sequence[dict[str, Any]], canonical_execs: Sequence[str], threshold: float = 0.25, n_seeds: int = 20, max_budget: int = 500) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deterministic = ["margin_state_major", "margin_breadth"]
    stochastic = [
        "margin_random_ties", "margin_fail_expand", "boundary_uniform", "boundary_map_elites", "boundary_only",
        "bc_qd", "bc_qd_no_boundary", "bc_qd_no_coverage", "bc_qd_no_novelty", "bc_qd_static_equal", "bc_qd_cumulative_credit",
    ]
    metrics = []
    checkpoints = []
    budgets = [25, 50, 100, 250, 500]
    base = 940000

    def consume(method: str, seed: int):
        hist, met = run_method(rows, method, seed, max_budget, threshold, canonical_execs)
        metrics.append(met)
        for b in budgets:
            h = hist[:min(b, len(hist))]
            checkpoints.append({"method": method, "seed": seed, "requested_budget": b, **_metrics(h)})

    for method in deterministic:
        consume(method, base)
    for i in range(n_seeds):
        for method in stochastic:
            consume(method, base + i)
    return metrics, checkpoints


def aggregate(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by = defaultdict(list)
    for row in metrics:
        by[row["method"]].append(row)
    out = {}
    for method, vals in by.items():
        keys = [k for k, v in vals[0].items() if isinstance(v, (int, float)) and k not in ("seed",)]
        out[method] = {}
        for key in keys:
            xs = [float(v[key]) for v in vals]
            out[method][key] = {"mean": statistics.mean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0}
    return out
