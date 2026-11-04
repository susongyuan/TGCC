# TGCC — hotdog (`cook_hot_dogs`)

Three-graph conditioning, success-distribution scoring and reference-conditioned π0.5 action generation for the BEHAVIOR **hotdog** task. The latest checkpoint is **v3fast step02153**, stopped by the user on 2026-11-04 before the planned 4000 joint updates. Its independent20-instance evaluation completed: **0/20 successes, mean officialQ=0.0000**.

The vision tower/projector and PaliGemma VLM parameters were **frozen** in this run. ActionExpert, action I/O, graphs, semantic/memory modules, scoring heads, retrieval adapter and training auxiliary heads were updated. The independent retrieval encoder stays frozen. This is not the earlier fully unfrozen TGCC-F run. There is no teacher–student distillation; `student` is an implementation member name.

|Release|Task|Actual training|Complete official20 result|
|---|---|---|---|
|[TC-π0.5](https://huggingface.co/Songyuan666/TC-pi0.5)|hotdog fine-tuning; separate unchanged50-task PT50 base|4970 formal×48, selected cumulative step5000|9/20, meanQ .5000|
|[TGCC-P](https://huggingface.co/Songyuan666/TGCC-P)|hotdog|1000 critic warmup+4708 partial joint; critic5708|10/20, meanQ .5000|
|[TGCC-F-hotdog-step05000](https://huggingface.co/Songyuan666/TGCC-F-hotdog-step05000)|hotdog|5000 fully unfrozen joint; inherited critic5708→10708|11/20, meanQ .5500|
|[TGCC-hotdog-v3fast-step02153](https://huggingface.co/Songyuan666/TGCC-hotdog-v3fast-step02153)|hotdog|2153 joint; vision/VLM frozen; critic2656→4809; policy source10000→12153|0/20, meanQ 0.0000|

These are different checkpoints and recipes; results are not interchangeable. All comparison suites use instances301–320, action seed42, environment seed0 and official success/Q. TC/TGCC use task-scoped **Oracle graphs** plus RGB. PT50 is the unchanged multi-task RGB baseline, not hotdog-only fine-tuning. No unfamiliar-task generalization is established. Historical source and model limitations remain in each pinned release.

## Files

- `frozen/`: exact latest training/inference source. `reproduction/frozen_source_sha256.json` binds200 checkpoint-owned files. Tests and upstream license accompany them.
- `runtime_extensions/`: explicit admission for the user-stopped checkpoint; preserves original `training_complete=false`, budget and weight bytes.
- `tools/`: source/asset verification, isolated-environment preparation and a portable latest-checkpoint inference launcher.
- Hugging Face `checkpoint/`, `retrieval_library/`, `source/`, `evaluation/`: weights, matched support/encoder, frozen source and result evidence. Older checkpoints require their own bundled source.

## Inference

Tested environment: Linux x86_64, Python3.11, PyTorch2.7.1+cu126, Transformers4.53.2. Isaac/BEHAVIOR assets are external. Use an isolated environment; `requirements.txt` pins the existing dependency stack. A GPU server needs no simulator. New portable commands have source/argument verification; a fresh third-party installation has not yet been validated for this release.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/prepare_env.py
python tools/download_weights.py --output weights
python tools/verify.py --assets weights/hotdog-v3fast
CUDA_VISIBLE_DEVICES=0 python tools/serve_hotdog.py \
  --assets weights/hotdog-v3fast --base-assets weights/tc-base \
  --output runs/hotdog-server --port 18835
```

Pin published Hub revisions for reproducibility. The service uses original32×23 action blocks, advisory scores, native controller limits, finite-value/interface checks and a durable execution protocol. It requires the original RGB/proprioception/Oracle-graph observation schema; this is not a Transformers AutoModel endpoint.

## Training and evaluation boundaries

Latest recipe: effective batch48 (4ranks×micro4×accum3),4 real burn-in blocks and8 supervised memory blocks, corrected actual-after-minus-before targets and reference-conditioning supervision. The original planned4000 updates remain in configuration; actual2153 updates and103344 window exposures remain in metadata. Saved model/Adam/RNG continuation is retained on the owner's server; public inference artifacts do not include a resumable optimizer checkpoint.

`correction_head` and `memory_probe` are training-only auxiliary loss heads; they do not directly emit online control commands. Online assistance uses trusted feedback, retrieved context and ActionExpert action generation. Triggering this path is distinct from successful recovery. The read-only [instance301 diagnosis](reproduction/diagnosis301/review.md) records64 assistance triggers and a repeated food-pickup stall; this is one case, not the complete20 result.

Calibration uses16 held-out demonstration episodes, excluding public test labels. A320-step interface/native-limit preflight is separate from the20 scored trials. Scoring does not terminate a task early. Only official natural terminals enter comparison tables; infrastructure interruptions remain unknown. No videos are recorded. Fixed texture streaming budget .05 preserves resolution and physics but does not guarantee identical resident texture mips.

The source includes historical experiment utilities. Training tools retain their original deployment/data assumptions; this release does not claim one-command portable training. No training or GPU benchmarking is started by these instructions.

## Attribution

Upstream OpenPI-Comet: `mli0603/openpi-comet`, commit `4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5`; license in `frozen/code/LICENSE`. Base: `sunshk/openpi_comet_pytorch` revision `5a489fe50fdc8e256533c8c611dfd60c247520cf`. Demonstrations: `behavior-1k/2025-challenge-demos`. Original code, model, dataset and scene-asset terms remain applicable. BEHAVIOR/Isaac and scene assets are not bundled.
