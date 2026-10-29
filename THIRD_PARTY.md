# Third-party assets

The repository does not redistribute model checkpoints or benchmark prompt text. They are loaded from the pinned upstream revisions in `paper_manifest.json`.

- `openai/gsm8k` — revision `740312add88f781978c0658806c59bc2815b9866`, MIT.
- `HuggingFaceH4/MATH-500` — revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`; see the upstream dataset card for terms.
- `openai/openai_humaneval` — revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, MIT.
- `google-research-datasets/mbpp` — revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, CC-BY-4.0.
- `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, and `HuggingFaceTB/SmolLM2-1.7B` are downloaded from the exact revisions listed in `paper_manifest.json`; their upstream licenses apply.

The bundled `reference_results/` directory contains experiment outputs derived from these public assets and is provided for result verification.
