from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from analysis import analyze, reproduce_reference_figure, validate_reference_search, validate_results
from experiment import ModelEvaluator, Prompt, decision_diagnostics, load_prompts, prepare_prompts, prepare_unseen_prompts, read_jsonl, write_json, write_jsonl
from search import run_method


MODELS = {
    "qwen0p6": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "qwen1p7": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "smollm2_1p7": ("HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71"),
}

CONDITIONS = {
    "qwen0p6_bf16_discovery": {"model": "qwen0p6", "dtype": "bfloat16", "prompt": "prompts.json", "split": "discovery", "context_seed": 20260911, "expected": 5016, "repeat": 3},
    "qwen0p6_bf16_heldout": {"model": "qwen0p6", "dtype": "bfloat16", "prompt": "prompts.json", "split": "heldout", "context_seed": 20260911, "expected": 1848, "repeat": 3},
    "qwen0p6_bf16_new_prompts": {"model": "qwen0p6", "dtype": "bfloat16", "prompt": "unseen_prompts.json", "split": "confirmatory", "context_seed": 20260912, "expected": 4224, "repeat": 3},
    "qwen1p7_bf16_discovery": {"model": "qwen1p7", "dtype": "bfloat16", "prompt": "prompts.json", "split": "discovery", "context_seed": 20260911, "expected": 5016, "repeat": 3},
    "qwen0p6_fp16_discovery": {"model": "qwen0p6", "dtype": "float16", "prompt": "prompts.json", "split": "discovery", "context_seed": 20260911, "expected": 5016, "repeat": 2},
    "smollm2_1p7b_bf16": {"model": "smollm2_1p7", "dtype": "bfloat16", "prompt": "prompts.json", "split": "discovery", "context_seed": 20260911, "expected": 5016, "repeat": 3},
}

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "paper_manifest.json"
REFERENCE = ROOT / "reference_results"


def parse_gpus(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out or len(out) != len(set(out)):
        raise ValueError("--gpus must contain unique physical GPU indices")
    return out


def shard_parents(prompts: list[Prompt], gpus: list[int]) -> dict[int, list[str]]:
    out = {g: [] for g in gpus}
    for i, prompt in enumerate(sorted(prompts, key=lambda x: (x.source, x.parent_id))):
        out[gpus[i % len(gpus)]].append(prompt.parent_id)
    return out


def worker(args) -> None:
    cfg = CONDITIONS[args.condition]
    model_id, revision = MODELS[cfg["model"]]
    all_prompts = load_prompts(args.prompt, cfg["split"])
    assigned = set(json.loads(Path(args.assignment).read_text(encoding="utf-8")))
    prompts = [p for p in all_prompts if p.parent_id in assigned]
    if not prompts:
        raise RuntimeError("empty worker shard")
    evaluator = ModelEvaluator(model_id, revision, cfg["dtype"], prompts, all_prompts, args.output, args.seed, cfg["context_seed"])
    try:
        evaluator.probe()
        if args.repeat > 0:
            write_json(evaluator.repeatability(args.repeat), Path(args.output) / "repeatability.json")
        evaluator.build_reference_cache(Path(args.output) / "reference")
        evaluator.evaluate_all(Path(args.output) / "records.jsonl", args.condition)
        rows = read_jsonl(Path(args.output) / "records.jsonl")
        write_json({"condition": args.condition, "parents": len(prompts), "states": len(evaluator.states), "pairs": len(rows)}, Path(args.output) / "summary.json")
    finally:
        evaluator.close()


def run_condition(name: str, root: Path, gpus: list[int]) -> None:
    matrix = root / "matrices" / f"{name}.jsonl"
    if matrix.exists() and len(read_jsonl(matrix)) == CONDITIONS[name]["expected"]:
        print(f"{name}: matrix already complete", flush=True)
        return

    cfg = CONDITIONS[name]
    prompt_path = root / "data" / cfg["prompt"]
    prompts = load_prompts(prompt_path, cfg["split"])
    shards = shard_parents(prompts, gpus)
    work = root / "workers" / name
    work.mkdir(parents=True, exist_ok=True)
    procs = []
    logs = []
    for i, gpu in enumerate(gpus):
        assign = work / f"gpu{gpu}_parents.json"
        write_json(shards[gpu], assign)
        out = work / f"gpu{gpu}"
        out.mkdir(parents=True, exist_ok=True)
        log = (work / f"gpu{gpu}.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        })
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "worker",
            "--condition", name,
            "--prompt", str(prompt_path),
            "--assignment", str(assign),
            "--output", str(out),
            "--seed", str(20260911 + i),
            "--repeat", str(cfg["repeat"] if i == 0 else 0),
        ]
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
        logs.append(log)

    while any(proc.poll() is None for _, proc in procs):
        live = [gpu for gpu, proc in procs if proc.poll() is None]
        print(f"{name}: running on physical GPUs {live}", flush=True)
        time.sleep(20)

    for log in logs:
        log.close()
    failed = [(gpu, proc.returncode) for gpu, proc in procs if proc.returncode != 0]
    if failed:
        raise RuntimeError(f"{name} failed: {failed}; inspect {work}/*.log")

    rows = []
    for gpu in gpus:
        rows.extend(read_jsonl(work / f"gpu{gpu}" / "records.jsonl"))
    rows.sort(key=lambda r: (r["source"], r["parent_id"], r["prefix_len"], r["execution"]))
    if len(rows) != cfg["expected"]:
        raise RuntimeError(f"{name}: expected {cfg['expected']} pairs, got {len(rows)}")
    if len({(r["state_id"], r["execution"]) for r in rows}) != len(rows):
        raise RuntimeError(f"{name}: duplicate state/execution rows")
    write_jsonl(rows, matrix)
    print(f"{name}: {len(rows)} pairs complete", flush=True)


def check_environment(require_cuda: bool = False) -> dict[str, str]:
    if sys.version_info < (3, 12):
        raise RuntimeError(f"Python 3.12+ required, found {sys.version.split()[0]}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["environment"]
    observed = {"python": sys.version.split()[0]}
    packages = ["numpy", "scipy", "pandas", "matplotlib", "datasets", "huggingface_hub", "accelerate", "transformers", "safetensors"]
    for name in packages:
        module = __import__(name)
        observed[name] = getattr(module, "__version__", "unknown")
        if observed[name] != expected[name]:
            raise RuntimeError(f"{name} version mismatch: {observed[name]} != {expected[name]}")
    import torch
    observed["torch"] = torch.__version__
    observed["cuda"] = str(torch.version.cuda)
    if observed["torch"] != expected["torch"]:
        raise RuntimeError(f"torch version mismatch: {observed['torch']} != {expected['torch']}")
    if require_cuda and (not torch.cuda.is_available() or str(torch.version.cuda) != "13.0"):
        raise RuntimeError(f"CUDA 13.0 build with a visible GPU required; torch CUDA={torch.version.cuda}")
    return observed


def preflight(gpus: list[int]) -> None:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], text=True)
    except Exception as exc:
        raise RuntimeError("nvidia-smi is required for the publication run") from exc
    available = {int(line.split(",", 1)[0].strip()) for line in out.splitlines() if line.strip()}
    missing = [g for g in gpus if g not in available]
    if missing:
        raise RuntimeError(f"requested GPUs not present: {missing}; available={sorted(available)}")
    print(out, flush=True)


def verify_prompt_ids(prompts_path: Path, unseen_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["expected_parent_ids"]
    payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    for split in ("discovery", "development", "heldout"):
        got = sorted(p["parent_id"] for p in payload["prompts"] if p["split"] == split)
        if got != expected["original"][split]:
            raise RuntimeError(f"pinned dataset verification failed for original {split} parent IDs")
    unseen = json.loads(unseen_path.read_text(encoding="utf-8"))
    got = sorted(p["parent_id"] for p in unseen["prompts"])
    if got != expected["unseen"]:
        raise RuntimeError("pinned dataset verification failed for unseen parent IDs")


def verify_datasets(args) -> None:
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prompts = out / "prompts.json"
    unseen = out / "unseen_prompts.json"
    prepare_prompts(prompts)
    prepare_unseen_prompts(prompts, unseen)
    verify_prompt_ids(prompts, unseen)
    print("pinned dataset revisions reproduce the paper parent IDs")


def reproduce(args) -> None:
    check_environment(require_cuda=True)
    root = Path(args.results).resolve()
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "matrices").mkdir(parents=True, exist_ok=True)
    gpus = parse_gpus(args.gpus)
    preflight(gpus)

    prompts = root / "data" / "prompts.json"
    unseen = root / "data" / "unseen_prompts.json"
    if not prompts.exists():
        prepare_prompts(prompts)
    if not unseen.exists():
        prepare_unseen_prompts(prompts, unseen)
    verify_prompt_ids(prompts, unseen)

    for name in CONDITIONS:
        run_condition(name, root, gpus)

    result = analyze(root, MANIFEST, args.search_seeds, validate=True)
    print(json.dumps(result["paper_validation"]["totals"], indent=2))
    print(f"results: {root / 'paper_results.json'}")


def verify_reference(args) -> None:
    result = validate_results(REFERENCE, MANIFEST, check_hashes=True)
    search = validate_reference_search(REFERENCE, MANIFEST)
    report = {"matrices_and_diagnostics": result, "published_search": search}
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reproduce_reference_figure(REFERENCE, MANIFEST, out / "main_geometry_search.png")
    write_json(report, out / "verification_report.json")
    if not result["pass"] or not search["pass"]:
        failures = result["failures"] + search["failures"]
        raise RuntimeError("reference validation failed: " + "; ".join(failures[:12]))
    print(json.dumps(result["totals"], indent=2))
    print("reference results match the camera-ready paper")
    print(f"figure: {out / 'main_geometry_search.png'}")


def self_test() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    from experiment import SOURCES
    for spec in SOURCES:
        expected = manifest["datasets"][spec["name"]]
        for key in ("repo", "revision", "config", "split", "field"):
            assert spec.get(key) == expected.get(key)
    for key, (repo, revision) in MODELS.items():
        assert manifest["models"][key]["repo"] == repo
        assert manifest["models"][key]["revision"] == revision

    ref = np.array([1.0, 0.9, -0.2], dtype=np.float32)
    tie = np.array([1.0, 1.0, -0.2], dtype=np.float32)
    d = decision_diagnostics(ref, tie, 0, 1)
    assert not d["strict_flip"]
    strict = np.array([1.0, 1.1, -0.2], dtype=np.float32)
    d = decision_diagnostics(ref, strict, 0, 1)
    assert d["strict_flip"] and d["strict_superiority"] and d["eta_condition"]
    # A competitor at a lower vocabulary index wins an exact alternative tie.
    ref_tie = np.array([0.9, 1.0, -0.2], dtype=np.float32)
    alt_tie = np.array([1.0, 1.0, -0.2], dtype=np.float32)
    d = decision_diagnostics(ref_tie, alt_tie, 1, 0)
    assert d["strict_flip"] and d["alternative_tie"] and not d["strict_superiority"]
    from analysis import _flip_plot_class
    assert _flip_plot_class({"strict_flip": True, "alt_winner_advantage": 0.0}) == "tie"
    assert _flip_plot_class({"strict_flip": True, "alt_winner_advantage": 0.125}) == "strict"

    rows = []
    executions = ["e0", "e1", "e2"]
    for state in range(40):
        margin = 0.05 if state < 8 else 0.5
        for i, execution in enumerate(executions):
            flip = state < 8 and i in (0, 1)
            rows.append({
                "state_id": f"s{state}", "parent_id": f"p{state // 2}", "source": "x", "prefix_len": 128,
                "execution": execution, "perturbation_family": "f", "shape_regime": execution,
                "ref_margin": margin, "strict_flip": flip, "alt_is_ref_top2": flip,
                "max_abs_delta": 0.1 if flip else 0.01, "risk_score": 4.0 if flip else 0.04,
            })
    for method in ("margin_fail_expand", "bc_qd"):
        history, metrics = run_method(rows, method, 7, 24, 0.25, executions)
        assert metrics["budget"] == 24 and len({(r["state_id"], r["execution"]) for r in history}) == 24

    rng = random.Random(123)
    for _ in range(10000):
        ref = np.array([rng.uniform(-2, 2) for _ in range(16)], dtype=np.float32)
        top = np.argsort(ref)[-2:][::-1]
        if ref[top[0]] == ref[top[1]]:
            continue
        alt = ref + np.array([rng.uniform(-0.2, 0.2) for _ in range(16)], dtype=np.float32)
        diag = decision_diagnostics(ref, alt, int(top[0]), int(top[1]))
        if diag["strict_flip"]:
            assert diag["eta_condition"]
    print("self-test passed")


def analyze_only(args) -> None:
    result = analyze(args.results, MANIFEST, args.search_seeds, validate=not args.no_validate)
    print(json.dumps(result.get("paper_validation", {}), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    reproduce_parser = sub.add_parser("reproduce")
    reproduce_parser.add_argument("--results", default="results")
    reproduce_parser.add_argument("--gpus", default="0,1,2,3")
    reproduce_parser.add_argument("--search-seeds", type=int, default=20)

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--results", default="results")
    analyze_parser.add_argument("--search-seeds", type=int, default=20)
    analyze_parser.add_argument("--no-validate", action="store_true")

    verify_parser = sub.add_parser("verify-reference")
    verify_parser.add_argument("--output", default="reference_reproduction")

    dataset_parser = sub.add_parser("verify-datasets")
    dataset_parser.add_argument("--output", default="dataset_verification")

    sub.add_parser("check-environment")
    sub.add_parser("self-test")

    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    worker_parser.add_argument("--prompt", required=True)
    worker_parser.add_argument("--assignment", required=True)
    worker_parser.add_argument("--output", required=True)
    worker_parser.add_argument("--seed", required=True, type=int)
    worker_parser.add_argument("--repeat", default=0, type=int)

    args = parser.parse_args()
    if args.cmd == "worker":
        worker(args)
    elif args.cmd == "reproduce":
        reproduce(args)
    elif args.cmd == "analyze":
        analyze_only(args)
    elif args.cmd == "verify-reference":
        verify_reference(args)
    elif args.cmd == "verify-datasets":
        verify_datasets(args)
    elif args.cmd == "check-environment":
        print(json.dumps(check_environment(require_cuda=False), indent=2))
    else:
        self_test()


if __name__ == "__main__":
    main()
