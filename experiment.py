from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Prompt:
    parent_id: str
    source: str
    text: str
    split: str


@dataclass(frozen=True)
class StateSpec:
    state_id: str
    parent_id: str
    source: str
    prefix_len: int
    parent_index: int
    ref_margin: float = 0.0
    ref_top1: int = -1
    ref_top2: int = -1


@dataclass(frozen=True)
class ExecutionSpec:
    name: str
    family: str
    mode: str
    chunk_size: int = 0
    batch_size: int = 1
    sdpa_backend: str = "auto"
    attn_impl: str = "sdpa"

    @property
    def shape_regime(self) -> str:
        if self.mode == "packed":
            return "packed"
        if self.mode == "chunked":
            return f"chunk{self.chunk_size}"
        if self.mode == "batch":
            return f"batch{self.batch_size}"
        return self.mode


SOURCES = [
    {"name": "gsm8k", "repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866", "config": "main", "split": "test", "field": "question"},
    {"name": "math500", "repo": "HuggingFaceH4/MATH-500", "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "config": None, "split": "test", "field": "problem"},
    {"name": "humaneval", "repo": "openai/openai_humaneval", "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544", "config": None, "split": "test", "field": "prompt"},
    {"name": "mbpp", "repo": "google-research-datasets/mbpp", "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "config": "full", "split": "test", "field": "text"},
]

EXECUTIONS = [
    ExecutionSpec("packed_auto", "token_shape", "packed", sdpa_backend="auto", attn_impl="sdpa"),
    ExecutionSpec("packed_math", "attention_kernel", "packed", sdpa_backend="math", attn_impl="sdpa"),
    ExecutionSpec("packed_flash", "attention_kernel", "packed", sdpa_backend="flash", attn_impl="sdpa"),
    ExecutionSpec("packed_eager", "attention_impl", "packed", sdpa_backend="auto", attn_impl="eager"),
    ExecutionSpec("chunk2_auto", "token_shape", "chunked", chunk_size=2),
    ExecutionSpec("chunk4_auto", "token_shape", "chunked", chunk_size=4),
    ExecutionSpec("chunk8_auto", "token_shape", "chunked", chunk_size=8),
    ExecutionSpec("chunk16_auto", "token_shape", "chunked", chunk_size=16),
    ExecutionSpec("chunk32_auto", "token_shape", "chunked", chunk_size=32),
    ExecutionSpec("batch2_auto", "batch_shape", "batch", batch_size=2),
    ExecutionSpec("batch4_auto", "batch_shape", "batch", batch_size=4),
]

PREFIX_LENGTHS = [64, 128, 256, 512, 768, 1024]


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _clean_text(x: Any) -> str:
    return "" if x is None else str(x).strip().replace("\x00", " ")


def _parent_id(source: str, raw_index: int | str, text: str) -> str:
    d = hashlib.sha1(f"{source}::{raw_index}::{text}".encode()).hexdigest()[:12]
    return f"{source}:{d}"


def _load_source(spec: dict[str, Any], n: int, seed: int) -> list[dict[str, str]]:
    from datasets import load_dataset

    kwargs = {}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    ds = load_dataset(spec["repo"], revision=spec["revision"], split=spec["split"], **kwargs)
    idx = list(range(len(ds)))
    salt = int(hashlib.sha1(spec["name"].encode()).hexdigest()[:8], 16)
    random.Random(seed ^ salt).shuffle(idx)
    rows = []
    for i in idx:
        text = _clean_text(ds[int(i)].get(spec["field"]))
        if not text:
            continue
        rows.append({"source": spec["name"], "text": text, "raw_index": str(i)})
        if len(rows) == n:
            break
    if len(rows) != n:
        raise RuntimeError(f"{spec['name']}: expected {n} usable rows, got {len(rows)}")
    return rows


def prepare_prompts(out_path: str | Path, seed: int = 20260911, n_per_source: int = 32) -> dict[str, Any]:
    from collections import defaultdict

    rows = []
    for spec in SOURCES:
        rows.extend(_load_source(spec, n_per_source, seed))
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    prompts = []
    for source in sorted(by_source):
        group = by_source[source]
        salt = int(hashlib.sha1(source.encode()).hexdigest()[:8], 16)
        random.Random(seed ^ salt).shuffle(group)
        n = len(group)
        nd = max(1, int(n * 0.60))
        nv = max(1, int(n * 0.20))
        if nd + nv >= n:
            nv = max(1, n - nd - 1)
        for i, row in enumerate(group):
            split = "discovery" if i < nd else "development" if i < nd + nv else "heldout"
            prompts.append(Prompt(_parent_id(row["source"], row["raw_index"], row["text"]), row["source"], row["text"], split))
    prompts.sort(key=lambda p: (p.split, p.source, p.parent_id))
    payload = {
        "metadata": {"seed": seed, "n_per_source": n_per_source, "num_prompts": len(prompts), "datasets": [{k: v for k, v in spec.items() if k != "field"} for spec in SOURCES]},
        "prompts": [asdict(p) for p in prompts],
    }
    write_json(payload, out_path)
    return payload


def prepare_unseen_prompts(prior_path: str | Path, out_path: str | Path, seed: int = 20260912, n_per_source: int = 16) -> dict[str, Any]:
    from datasets import load_dataset

    prior = json.loads(Path(prior_path).read_text(encoding="utf-8"))
    old_ids = {x["parent_id"] for x in prior["prompts"]}
    old_hashes = {hashlib.sha256(x["text"].encode()).hexdigest() for x in prior["prompts"]}
    prompts = []
    for spec in SOURCES:
        kwargs = {}
        if spec.get("config"):
            kwargs["name"] = spec["config"]
        ds = load_dataset(spec["repo"], revision=spec["revision"], split=spec["split"], **kwargs)
        idx = list(range(len(ds)))
        salt = int(hashlib.sha1(spec["name"].encode()).hexdigest()[:8], 16)
        random.Random(seed ^ salt).shuffle(idx)
        got = 0
        for i in idx:
            text = _clean_text(ds[int(i)].get(spec["field"]))
            if not text:
                continue
            pid = _parent_id(spec["name"], i, text)
            th = hashlib.sha256(text.encode()).hexdigest()
            if pid in old_ids or th in old_hashes:
                continue
            prompts.append(Prompt(pid, spec["name"], text, "confirmatory"))
            got += 1
            if got == n_per_source:
                break
        if got != n_per_source:
            raise RuntimeError(f"{spec['name']}: expected {n_per_source} unseen prompts, got {got}")
    prompts.sort(key=lambda p: (p.source, p.parent_id))
    payload = {
        "metadata": {"seed": seed, "n_per_source": n_per_source, "num_prompts": len(prompts), "prior_parent_overlap": 0, "datasets": [{k: v for k, v in spec.items() if k != "field"} for spec in SOURCES]},
        "prompts": [asdict(p) for p in prompts],
    }
    write_json(payload, out_path)
    return payload


def load_prompts(path: str | Path, split: str | None = None) -> list[Prompt]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = [Prompt(**x) for x in payload["prompts"]]
    if split and split != "all":
        prompts = [p for p in prompts if p.split == split]
    return prompts


def build_context_token_ids(tokenizer: Any, prompts: Sequence[Prompt], max_tokens: int, seed: int) -> list[list[int]]:
    eos = getattr(tokenizer, "eos_token_id", None)
    sep_ids = [int(eos)] if eos is not None else list(tokenizer.encode("\n\n", add_special_tokens=False))
    if not sep_ids:
        sep_ids = [0]
    encoded = [list(map(int, tokenizer.encode(p.text, add_special_tokens=False))) for p in prompts]
    contexts = []
    for i, p in enumerate(prompts):
        ids = list(encoded[i]) or sep_ids[:1]
        digest = int(hashlib.sha1(f"{seed}:{p.parent_id}".encode()).hexdigest()[:16], 16)
        order = list(range(len(prompts)))
        random.Random(digest).shuffle(order)
        cursor = 0
        while len(ids) < max_tokens:
            donor = encoded[order[cursor % len(order)]]
            cursor += 1
            if donor:
                ids.extend(sep_ids)
                ids.extend(donor)
        contexts.append(ids[:max_tokens])
    return contexts


def make_states(prompts: Sequence[Prompt], contexts: Sequence[Sequence[int]]) -> list[StateSpec]:
    out = []
    for pi, (p, ids) in enumerate(zip(prompts, contexts)):
        for plen in PREFIX_LENGTHS:
            if plen <= len(ids):
                out.append(StateSpec(f"{p.parent_id}::L{plen}", p.parent_id, p.source, plen, pi))
    return out


def _top2(arr: np.ndarray) -> tuple[int, int]:
    idx = np.argpartition(arr, -2)[-2:]
    idx = idx[np.argsort(arr[idx])[::-1]]
    return int(idx[0]), int(idx[1])


def decision_diagnostics(ref: np.ndarray, alt: np.ndarray, ref_top1: int, ref_top2: int) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float32)
    alt = np.asarray(alt, dtype=np.float32)
    w, t2 = int(ref_top1), int(ref_top2)
    alt_top1 = int(np.argmax(alt))
    margin = float(ref[w] - ref[t2])
    d = alt - ref
    d_w = float(d[w])
    rel = d - d_w
    rel2 = rel.copy()
    rel2[w] = -np.inf
    eta_token = int(np.argmax(rel2))
    eta = float(rel2[eta_token])
    max_abs = float(np.max(np.abs(d)))
    alt_adv = float(alt[alt_top1] - alt[w])
    flip = bool(alt_top1 != w and margin > 0.0)
    return {
        "alt_top1": alt_top1,
        "max_abs_delta": max_abs,
        "two_linf": 2.0 * max_abs,
        "max_competitor_relative_shift": eta,
        "max_competitor_shift_token": eta_token,
        "eta_to_margin_ratio": float(eta / (margin + 1e-12)) if margin > 0 else float("inf"),
        "linf_to_margin_ratio": float(2.0 * max_abs / (margin + 1e-12)) if margin > 0 else float("inf"),
        "alt_winner_advantage": alt_adv,
        "alternative_tie": bool(flip and alt_adv == 0.0),
        "strict_superiority": bool(flip and alt_adv > 0.0),
        "eta_condition": bool(eta + 1e-12 >= margin),
        "strict_flip": flip,
        "alt_is_ref_top2": bool(alt_top1 == t2),
    }


class ModelEvaluator:
    def __init__(self, model_id: str, revision: str, dtype_name: str, prompts: Sequence[Prompt], full_prompts: Sequence[Prompt], work_dir: str | Path, seed: int, context_seed: int):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.transformers = transformers
        self.model_id = model_id
        self.revision = revision
        self.dtype_name = dtype_name
        self.prompts = list(prompts)
        self.full_prompts = list(full_prompts)
        self.seed = int(seed)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("worker must see exactly one CUDA GPU")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=False)
        self.device = torch.device("cuda:0")
        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtypes[dtype_name]

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        kwargs = {"revision": revision, "trust_remote_code": False, "attn_implementation": "sdpa"}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype, **kwargs)
        self.model.to(self.device).eval()
        self.current_attn_impl = "sdpa"
        self.forward_params = set(inspect.signature(self.model.forward).parameters)

        full_contexts = build_context_token_ids(self.tokenizer, self.full_prompts, 1024, context_seed)
        by_parent = {p.parent_id: ids for p, ids in zip(self.full_prompts, full_contexts)}
        self.contexts = [by_parent[p.parent_id] for p in self.prompts]
        self.states = make_states(self.prompts, self.contexts)
        self.ref_logits: np.ndarray | None = None
        write_json({
            "model_id": model_id,
            "revision": revision,
            "resolved_model_commit": getattr(self.model.config, "_commit_hash", None),
            "dtype": str(self.dtype),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "num_prompts": len(self.prompts),
            "num_states": len(self.states),
        }, self.work_dir / "manifest.json")

    def _position_kwargs(self, batch: int, start: int, length: int) -> dict[str, Any]:
        t = self.torch
        pos = t.arange(start, start + length, device=self.device, dtype=t.long)
        out = {}
        if "position_ids" in self.forward_params:
            out["position_ids"] = pos.unsqueeze(0).expand(batch, -1)
        if "cache_position" in self.forward_params:
            out["cache_position"] = pos
        return out

    def _sdpa_context(self, backend: str):
        if backend == "auto" or self.current_attn_impl != "sdpa":
            return contextlib.nullcontext()
        from torch.nn.attention import SDPBackend, sdpa_kernel
        mapping = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}
        return sdpa_kernel(backends=[mapping[backend]])

    def _set_attn(self, impl: str) -> None:
        if impl == self.current_attn_impl:
            return
        setter = getattr(self.model, "set_attn_implementation", None)
        if setter is None:
            raise RuntimeError("runtime attention implementation switching unavailable")
        setter(impl)
        self.current_attn_impl = impl

    def _forward(self, ids: Any, past: Any = None, start: int = 0) -> Any:
        kw = {"input_ids": ids, "use_cache": True, "return_dict": True}
        if past is not None:
            kw["past_key_values"] = past
        kw.update(self._position_kwargs(ids.shape[0], start, ids.shape[1]))
        return self.model(**kw)

    def serial_next_logits(self, tokens: Sequence[int]):
        t = self.torch
        past = None
        out = None
        self._set_attn("sdpa")
        with t.inference_mode(), self._sdpa_context("auto"):
            for i, tok in enumerate(tokens):
                ids = t.tensor([[int(tok)]], device=self.device, dtype=t.long)
                out = self._forward(ids, past=past, start=i)
                past = out.past_key_values
        return out.logits[0, -1].detach()

    def execute(self, tokens: Sequence[int], e: ExecutionSpec):
        t = self.torch
        prev = self.current_attn_impl
        try:
            self._set_attn(e.attn_impl)
            with t.inference_mode(), self._sdpa_context(e.sdpa_backend):
                if e.mode == "packed":
                    ids = t.tensor([list(map(int, tokens))], device=self.device, dtype=t.long)
                    return self._forward(ids, start=0).logits[0, -1].detach()
                if e.mode == "batch":
                    ids = t.tensor([list(map(int, tokens))], device=self.device, dtype=t.long).repeat(e.batch_size, 1)
                    return self._forward(ids, start=0).logits[0, -1].detach()
                past = None
                out = None
                for start in range(0, len(tokens), e.chunk_size):
                    chunk = t.tensor([list(map(int, tokens[start:start + e.chunk_size]))], device=self.device, dtype=t.long)
                    out = self._forward(chunk, past=past, start=start)
                    past = out.past_key_values
                return out.logits[0, -1].detach()
        finally:
            if self.current_attn_impl != prev:
                self._set_attn(prev)

    def probe(self) -> list[ExecutionSpec]:
        checks = []
        supported = []
        for e in EXECUTIONS:
            ok = True
            err = None
            for plen in (128, 1024):
                try:
                    _ = self.execute(self.contexts[0][:plen], e)
                    self.torch.cuda.synchronize()
                except Exception as exc:
                    ok = False
                    err = repr(exc)
                    self.torch.cuda.empty_cache()
                    break
            checks.append({"execution": asdict(e), "supported": ok, "error": err})
            if ok:
                supported.append(e)
        write_json(checks, self.work_dir / "execution_probe.json")
        if len(supported) != len(EXECUTIONS):
            missing = [x.name for x in EXECUTIONS if x.name not in {e.name for e in supported}]
            raise RuntimeError(f"unsupported execution regimes: {missing}")
        return supported

    def repeatability(self, n: int) -> list[dict[str, Any]]:
        n = min(n, len(self.states))
        if n <= 0:
            return []
        idx = [round(i * (len(self.states) - 1) / max(1, n - 1)) for i in range(n)]
        rows = []
        for i in idx:
            s = self.states[i]
            tokens = self.contexts[s.parent_index][:s.prefix_len]
            a = self.serial_next_logits(tokens).float().cpu().numpy()
            b = self.serial_next_logits(tokens).float().cpu().numpy()
            rows.append({
                "state_id": s.state_id,
                "top1_equal": int(np.argmax(a)) == int(np.argmax(b)),
                "bitwise_equal_logits": bool(np.array_equal(a, b)),
                "max_abs_delta": float(np.max(np.abs(a - b))),
            })
        return rows

    def build_reference_cache(self, cache_dir: str | Path) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vocab = int(self.model.config.vocab_size)
        logits_path = cache_dir / "reference_logits_f32.npy"
        mmap = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(self.states), vocab))
        row_by_parent_len = {(s.parent_index, s.prefix_len): i for i, s in enumerate(self.states)}
        updated: list[StateSpec | None] = [None] * len(self.states)
        checkpoints = set(PREFIX_LENGTHS)
        for pi, tokens in enumerate(self.contexts):
            past = None
            self._set_attn("sdpa")
            with self.torch.inference_mode(), self._sdpa_context("auto"):
                for pos in range(1024):
                    ids = self.torch.tensor([[int(tokens[pos])]], device=self.device, dtype=self.torch.long)
                    out = self._forward(ids, past=past, start=pos)
                    past = out.past_key_values
                    plen = pos + 1
                    if plen in checkpoints:
                        row = row_by_parent_len[(pi, plen)]
                        arr = out.logits[0, -1].detach().float().cpu().numpy()
                        mmap[row] = arr
                        t1, t2 = _top2(arr)
                        updated[row] = replace(self.states[row], ref_margin=float(arr[t1] - arr[t2]), ref_top1=t1, ref_top2=t2)
            print(f"reference {pi + 1}/{len(self.contexts)}", flush=True)
        mmap.flush()
        if any(x is None for x in updated):
            raise RuntimeError("reference cache incomplete")
        self.states = [x for x in updated if x is not None]
        self.ref_logits = np.load(logits_path, mmap_mode="r")
        write_json([asdict(x) for x in self.states], cache_dir / "states.json")

    def evaluate_all(self, out_path: str | Path, label: str) -> None:
        if self.ref_logits is None:
            raise RuntimeError("build reference cache first")
        rows = []
        for si, s in enumerate(self.states):
            tokens = self.contexts[s.parent_index][:s.prefix_len]
            ref = np.asarray(self.ref_logits[si])
            for e in EXECUTIONS:
                self.torch.cuda.synchronize()
                start = time.perf_counter()
                alt = self.execute(tokens, e)
                self.torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                alt_np = alt.float().cpu().numpy()
                d = decision_diagnostics(ref, alt_np, s.ref_top1, s.ref_top2)
                row = {
                    "label": label,
                    "state_id": s.state_id,
                    "parent_id": s.parent_id,
                    "source": s.source,
                    "prefix_len": s.prefix_len,
                    "execution": e.name,
                    "perturbation_family": e.family,
                    "shape_regime": e.shape_regime,
                    "ref_top1": s.ref_top1,
                    "ref_top2": s.ref_top2,
                    "ref_margin": s.ref_margin,
                    "risk_score": float(2.0 * d["max_abs_delta"] / (s.ref_margin + 1e-12)) if s.ref_margin > 0 else float("inf"),
                    "elapsed_ms": elapsed_ms,
                    **d,
                }
                rows.append(row)
            if (si + 1) % 20 == 0:
                print(f"{label}: states {si + 1}/{len(self.states)}", flush=True)
        write_jsonl(rows, out_path)

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass
