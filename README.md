# autoresearch-sdxl-lora

Systematic SDXL LoRA fine-tuning framework running on Modal cloud A100-80GB GPUs. Conducts controlled experiments varying LoRA rank and learning rate using Diffusers + PEFT, with tracking via W&B and a local results spreadsheet.

## Experimental Design

The first sweep isolates the effect of LoRA rank and learning rate on visual quality. All other variables are held constant:

- **Base model:** `stabilityai/stable-diffusion-xl-base-1.0`
- **VAE:** `madebyollin/sdxl-vae-fp16-fix` (fp16 NaN fix)
- **LoRA scope:** UNet-only (`to_q`, `to_k`, `to_v`, `to_out.0`)
- **Alpha:** always equals rank
- **Precision:** bf16
- **Steps:** 1000 per run, cosine LR schedule with 100 warmup steps
- **Effective batch size:** 4 (batch=1, gradient accumulation=4)
- **Validation:** 4 fixed prompts, seed 42

## Experiment Matrix

| Exp | Rank | Alpha | LR   | Status  |
|-----|------|-------|------|---------|
| 001 | 8    | 8     | 1e-4 | pending |
| 002 | 8    | 8     | 5e-5 | pending |
| 003 | 16   | 16    | 1e-4 | pending |
| 004 | 16   | 16    | 5e-5 | pending |
| 005 | 32   | 32    | 1e-4 | pending |
| 006 | 32   | 32    | 5e-5 | pending |
| 007 | 64   | 64    | 1e-4 | pending |
| 008 | 64   | 64    | 5e-5 | pending |

Top 2 runs from Sweep 1 advance to Sweep 2, which varies text encoder LoRA, step count, rank refinement, and alpha ratio.

## Quick Start

### Prerequisites

1. Install the Modal CLI and authenticate:
   ```bash
   pip install modal
   modal setup
   ```

2. Create the required secrets (one-time):
   ```bash
   modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
   modal secret create wandb-secret WANDB_API_KEY=xxxxx
   ```

### Running Experiments

```bash
# Run a single experiment (e.g., rank=16, lr=1e-4)
modal run modal_train.py --exp-id 003

# Run the full 8-experiment sweep
modal run modal_train.py --exp-id all

# Generate validation images from a trained LoRA
modal run modal_train.py --exp-id 003 --mode infer
```

## Project Structure

```
sdxl-lora-modal/
  modal_train.py          # Modal app with train() and infer() functions
  design.md               # Detailed experimental design and heuristics
  results.tsv             # Experiment tracking (8 rows)
  configs/                # Per-experiment YAML configs (planned)
  data/                   # Training images + captions
  prompts/
    validation.txt        # 4 fixed validation prompts
  outputs/                # Populated by Modal runs (LoRA weights, logs, images)
```

## Evaluation Strategy

**Quantitative** -- tracked per run via W&B:
- Training loss curve
- Validation loss at checkpoints

**Qualitative** -- compared across top runs:
- Same 4 prompts rendered at the same seed for side-by-side comparison
- Assessed for style adherence, prompt following, artifacts, and diversity

**Keep/discard rule:** keep if visually coherent with no artifacts; discard on mode collapse or heavy artifacts. Top 2 advance to Sweep 2.

## Tech Stack

| Component | Choice |
|-----------|--------|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| VAE | `madebyollin/sdxl-vae-fp16-fix` |
| Training | Diffusers `train_text_to_image_lora_sdxl.py` + PEFT |
| Compute | Modal A100-80GB (~$3.74/hr on-demand) |
| Precision | bf16 |
| Tracking | Weights & Biases |
| Results | `results.tsv` |

## Cost Estimate

- ~15-30 min per 1000-step run
- 8-run Sweep 1: ~$15-30
- With Sweep 2 (~6 additional runs): ~$30-50 total
