# EvoRobust 2026 execution-fragility experiments

Code and reference outputs for:

**Execution Fragility in LLM Inference: Margin–Displacement Geometry and Strong Baselines for Diversity-Driven Robustness Search**

The repository contains the six original result matrices used for the camera-ready paper, the enriched decision-directed diagnostic samples, the original search replay outputs, and the code required to regenerate them.

## Requirements

The paper run used:

- Python 3.12
- PyTorch 2.14.0+cu130
- Transformers 5.17.0
- NVIDIA A100-SXM4-80GB

Create the environment on Linux with a CUDA-13-capable NVIDIA driver:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python run.py check-environment
python run.py self-test
```

`requirements.txt` is pinned to the package versions used in the paper. PyTorch is installed separately because the CUDA wheel comes from the PyTorch `cu130` index.

## Verify the published results without GPUs

```bash
python run.py verify-reference
```

This checks SHA-256 hashes, all six matrix sizes, screen denominators, condition-level flip counts, the paper total of 195 flips, the 152 tie-resolved / 43 strict-superiority split, fragile-state/parent counts, diagnostic AUCs, and the published normalized search-AUC table. It also regenerates Figure 1 at:

```text
reference_reproduction/main_geometry_search.png
reference_reproduction/main_geometry_search.pdf
```

Expected terminal summary:

```text
{
  "pairs": 26136,
  "flips": 195,
  "tie_flips": 152,
  "strict_superiority_flips": 43
}
reference results match the camera-ready paper
```

The original matrices are under `reference_results/matrices/`. Their checksums are in `reference_results/SHA256SUMS.txt` and `paper_manifest.json`.

## Verify the pinned datasets

The four public datasets are loaded at immutable Hugging Face revisions recorded in `paper_manifest.json`:

```text
openai/gsm8k                    740312add88f781978c0658806c59bc2815b9866
HuggingFaceH4/MATH-500          6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
openai/openai_humaneval         7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544
google-research-datasets/mbpp   4bb6404fdc6cacfda99d4ac4205087b89d32030c
```

To download those exact revisions, rebuild both prompt draws, and assert that every generated parent ID matches the paper:

```bash
python run.py verify-datasets
```

No benchmark prompt text is redistributed in this repository.

## Reproduce the GPU experiments

On four A100s:

```bash
python run.py reproduce --gpus 0,1,2,3 --results results
```

Any comma-separated subset of physical GPU indices can be used. Each worker receives exactly one GPU through `CUDA_VISIBLE_DEVICES`.

The full run regenerates:

- Qwen3-0.6B BF16 discovery
- Qwen3-0.6B BF16 held-out
- Qwen3-0.6B BF16 unseen prompts
- Qwen3-1.7B BF16 discovery
- Qwen3-0.6B FP16 discovery
- SmolLM2-1.7B BF16 discovery

The exact model revisions are in `paper_manifest.json`.

Before GPU execution the code asserts that the pinned datasets reproduce the original parent IDs. After analysis it asserts that the new matrices reproduce the paper's condition-level counts and diagnostic statistics.

## Determinism contract

The publication run used:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=0
TOKENIZERS_PARALLELISM=false
```

Each model worker additionally uses:

```text
torch.manual_seed(...)
torch.cuda.manual_seed_all(...)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
model.eval()
```

The serial reference is batch size 1, one token at a time with KV cache, `attn_implementation="sdpa"`, PyTorch SDPA backend `auto`, explicit position/cache positions when supported, and TF32 disabled. The 11 alternative regimes are defined in `experiment.py`.

## Outputs

```text
results/matrices/*.jsonl
results/tables/screen_denominators.csv
results/tables/threshold_sensitivity.csv
results/tables/diagnostic_auc.csv
results/tables/normalized_search_auc.csv
results/tables/full_budget_trajectory.csv
results/tables/paired_wilcoxon.csv
results/tables/strict_superiority_search.csv
results/tables/repeatability.csv
results/figures/main_geometry_search.png
results/figures/main_geometry_search.pdf
results/paper_results.json
```

Analysis can be rerun without inference:

```bash
python run.py analyze --results results
```

## Reference-result layout

```text
reference_results/
  matrices/       original six complete matrices
  diagnostics/    original enriched full-logit diagnostic rows and sampling plans
  search/         original P4 search-replay outputs used in the paper
  SHA256SUMS.txt
```

The geometry panel derives its x/y limits from the bundled diagnostic data; for the paper data they are 15.0 and 2.5. The discovery repertoire size is computed from the matrix and is 28; it is not hardcoded by the plotting function.

## License

Code is released under the MIT License. Upstream datasets and model checkpoints retain their own licenses; see `THIRD_PARTY.md`.
