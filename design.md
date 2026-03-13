# SDXL LoRA Experiment Design — Modal

Systematic LoRA fine-tuning on Stable Diffusion XL using Diffusers + PEFT, deployed on Modal with A100 GPUs.

## Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` | Industry standard SDXL |
| VAE | `madebyollin/sdxl-vae-fp16-fix` | Fixed fp16 NaN issues |
| Trainer | `train_text_to_image_lora_sdxl.py` (Diffusers examples) | Official, PEFT-integrated |
| LoRA lib | PEFT via Diffusers | LoraConfig for rank/alpha/targets |
| Compute | Modal A100-80GB | On-demand, no infra management |
| Tracking | W&B (`--report_to wandb`) | TensorBoard fallback available |
| Precision | bf16 | Better dynamic range than fp16 on A100 |

## LoRA Scope

**First sweep: UNet only.** Text encoder LoRA is off. This isolates the effect of rank and LR on the visual prior without confounding text understanding changes.

Default target modules (Diffusers/PEFT default for SDXL UNet):
- `to_q`, `to_k`, `to_v`, `to_out.0` (attention projections)

## Project Structure

```
sdxl-lora-experiments/
├── modal_train.py          # Modal app: image, volume, train function
├── modal_infer.py          # Modal app: load LoRA, generate comparisons
├── configs/
│   ├── base.yaml           # Shared defaults
│   ├── exp_001.yaml        # rank=8, lr=1e-4
│   ├── exp_002.yaml        # rank=8, lr=5e-5
│   ├── exp_003.yaml        # rank=16, lr=1e-4
│   ├── exp_004.yaml        # rank=16, lr=5e-5
│   ├── exp_005.yaml        # rank=32, lr=1e-4
│   ├── exp_006.yaml        # rank=32, lr=5e-5
│   ├── exp_007.yaml        # rank=64, lr=1e-4
│   └── exp_008.yaml        # rank=64, lr=5e-5
├── data/
│   └── (your training images + captions)
├── prompts/
│   └── validation.txt      # 4-8 fixed validation prompts
├── outputs/                # Populated by Modal runs
│   ├── exp_001/
│   │   ├── pytorch_lora_weights.safetensors
│   │   └── logs/
│   └── ...
└── results.tsv             # Experiment tracking
```

## Baseline Config

```yaml
# configs/base.yaml
model_name: "stabilityai/stable-diffusion-xl-base-1.0"
vae_name: "madebyollin/sdxl-vae-fp16-fix"
resolution: 1024
mixed_precision: "bf16"
gradient_checkpointing: true
lr_scheduler: "cosine"
lr_warmup_steps: 100
train_batch_size: 1             # increase if VRAM allows
gradient_accumulation_steps: 4  # effective batch = 4
max_train_steps: 1000
checkpointing_steps: 250
seed: 42
rank: 16
alpha: 16                       # alpha = rank (standard)
caption_column: "text"
dataloader_num_workers: 4
```

## Sweep 1: Rank x Learning Rate (8 runs)

| Exp | Rank | Alpha | LR   | Everything else |
|-----|------|-------|------|-----------------|
| 001 | 8    | 8     | 1e-4 | base.yaml       |
| 002 | 8    | 8     | 5e-5 | base.yaml       |
| 003 | 16   | 16    | 1e-4 | base.yaml       |
| 004 | 16   | 16    | 5e-5 | base.yaml       |
| 005 | 32   | 32    | 1e-4 | base.yaml       |
| 006 | 32   | 32    | 5e-5 | base.yaml       |
| 007 | 64   | 64    | 1e-4 | base.yaml       |
| 008 | 64   | 64    | 5e-5 | base.yaml       |

**Fixed across all runs:** same captions, same resolution (1024), same validation prompts, same seeds, same step count (1000).

## Sweep 2 (after Sweep 1 analysis)

Take top 2 runs from Sweep 1, then vary:
- Text encoder LoRA: off vs on
- Training steps: 500 vs 1000 vs 2000
- Rank refinement: 12, 24, 48
- Alpha ratio: alpha=rank vs alpha=rank/2

## Decision Heuristics

- **Style LoRAs** tolerate higher LR (~1e-4) — style is a broad feature.
- **Identity/concept LoRAs** prefer lower LR (~5e-5) and better captions — identity is narrow.
- When in doubt, rank 16 + LR 1e-4 is the strongest starting point.
- If FID/CLIP scores plateau, improving captions > tuning hyperparams.

---

## Modal Implementation

### `modal_train.py`

```python
"""
SDXL LoRA training on Modal.

Usage:
    # Single experiment
    modal run modal_train.py --exp-id 001

    # Full sweep
    for i in $(seq -w 1 8); do modal run modal_train.py --exp-id $i; done
"""
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

MINUTES = 60

volume = modal.Volume.from_name("sdxl-lora-experiments", create_if_missing=True)
MODEL_DIR = "/model"
OUTPUT_DIR = "/outputs"
DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "diffusers==0.31.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "peft==0.13.2",
        "safetensors==0.4.4",
        "datasets==3.0.0",
        "huggingface-hub==0.36.0",
        "wandb==0.18.0",
        "Pillow>=10.0.0",
        "numpy<2",
        "pyyaml",
        "bitsandbytes==0.44.1",
    )
    .env({"HF_HUB_CACHE": "/cache", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_commands(
        # Pre-clone the Diffusers examples so the training script is available
        "git clone --depth 1 https://github.com/huggingface/diffusers.git /diffusers_repo"
    )
)

app = modal.App("sdxl-lora-training", image=image)

# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

SWEEP_1 = {
    "001": {"rank": 8,  "alpha": 8,  "learning_rate": 1e-4},
    "002": {"rank": 8,  "alpha": 8,  "learning_rate": 5e-5},
    "003": {"rank": 16, "alpha": 16, "learning_rate": 1e-4},
    "004": {"rank": 16, "alpha": 16, "learning_rate": 5e-5},
    "005": {"rank": 32, "alpha": 32, "learning_rate": 1e-4},
    "006": {"rank": 32, "alpha": 32, "learning_rate": 5e-5},
    "007": {"rank": 64, "alpha": 64, "learning_rate": 1e-4},
    "008": {"rank": 64, "alpha": 64, "learning_rate": 5e-5},
}

# Shared defaults
BASE_CONFIG = {
    "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
    "vae_name": "madebyollin/sdxl-vae-fp16-fix",
    "resolution": 1024,
    "mixed_precision": "bf16",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 100,
    "max_train_steps": 1000,
    "checkpointing_steps": 250,
    "seed": 42,
    "dataloader_num_workers": 4,
}

# ---------------------------------------------------------------------------
# Validation prompts (same across all runs)
# ---------------------------------------------------------------------------

VALIDATION_PROMPTS = [
    "a photo of a cat sitting on a windowsill at sunset",
    "an oil painting of a mountain landscape in autumn",
    "a digital illustration of a robot reading a book",
    "a watercolor portrait of an elderly woman smiling",
]


@app.function(
    gpu="A100-80GB",
    volumes={
        MODEL_DIR: volume,
        OUTPUT_DIR: volume,
        DATA_DIR: volume,
    },
    timeout=60 * MINUTES,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"]),
    ],
)
def train(exp_id: str, dataset_name: str = "lambdalabs/naruto-blip-captions"):
    """Run a single LoRA training experiment."""
    from accelerate.utils import write_basic_config

    # Merge base config with experiment-specific overrides
    if exp_id not in SWEEP_1:
        raise ValueError(f"Unknown experiment ID: {exp_id}. Valid: {list(SWEEP_1.keys())}")

    config = {**BASE_CONFIG, **SWEEP_1[exp_id]}
    exp_output_dir = f"{OUTPUT_DIR}/exp_{exp_id}"
    os.makedirs(exp_output_dir, exist_ok=True)

    # Set up accelerate for single-GPU bf16 training
    write_basic_config(mixed_precision="bf16")

    # Build the training command
    cmd = [
        "accelerate", "launch",
        "/diffusers_repo/examples/text_to_image/train_text_to_image_lora_sdxl.py",
        f"--pretrained_model_name_or_path={config['model_name']}",
        f"--pretrained_vae_model_name_or_path={config['vae_name']}",
        f"--dataset_name={dataset_name}",
        "--caption_column=text",
        f"--resolution={config['resolution']}",
        f"--train_batch_size={config['train_batch_size']}",
        f"--gradient_accumulation_steps={config['gradient_accumulation_steps']}",
        f"--learning_rate={config['learning_rate']}",
        f"--lr_scheduler={config['lr_scheduler']}",
        f"--lr_warmup_steps={config['lr_warmup_steps']}",
        f"--max_train_steps={config['max_train_steps']}",
        f"--checkpointing_steps={config['checkpointing_steps']}",
        f"--seed={config['seed']}",
        f"--output_dir={exp_output_dir}",
        f"--rank={config['rank']}",
        f"--mixed_precision={config['mixed_precision']}",
        f"--validation_prompt={VALIDATION_PROMPTS[0]}",
        "--validation_epochs=1",
        "--report_to=wandb",
    ]

    if config.get("gradient_checkpointing"):
        cmd.append("--gradient_checkpointing")

    print(f"=== Experiment {exp_id} ===")
    print(f"rank={config['rank']}, alpha={config['alpha']}, lr={config['learning_rate']}")
    print(f"Output: {exp_output_dir}")
    print(f"Command: {' '.join(cmd)}")

    # Run the training script as a subprocess (required by accelerate)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            print(line.decode(), end="")

    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"Training failed with exit code {exit_code}")

    # Commit volume so outputs persist
    volume.commit()
    print(f"=== Experiment {exp_id} complete. Outputs saved to {exp_output_dir} ===")
    return exp_output_dir


@app.function(
    gpu="A100-80GB",
    volumes={MODEL_DIR: volume, OUTPUT_DIR: volume},
    timeout=30 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def infer(exp_id: str, prompts: list[str] = None):
    """Generate validation images from a trained LoRA for comparison."""
    import torch
    from diffusers import DiffusionPipeline

    if prompts is None:
        prompts = VALIDATION_PROMPTS

    exp_output_dir = f"{OUTPUT_DIR}/exp_{exp_id}"
    results_dir = f"{exp_output_dir}/validation_images"
    os.makedirs(results_dir, exist_ok=True)

    # Load base SDXL + LoRA adapter
    pipe = DiffusionPipeline.from_pretrained(
        BASE_CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.load_lora_weights(exp_output_dir)

    generator = torch.Generator(device="cuda").manual_seed(BASE_CONFIG["seed"])

    for i, prompt in enumerate(prompts):
        image = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        image.save(f"{results_dir}/prompt_{i:02d}.png")
        print(f"Saved: prompt_{i:02d}.png — '{prompt}'")

    volume.commit()
    print(f"Validation images saved to {results_dir}")


@app.local_entrypoint()
def main(exp_id: str = "003", mode: str = "train", dataset_name: str = "lambdalabs/naruto-blip-captions"):
    """
    Entry point.

    Examples:
        modal run modal_train.py --exp-id 003
        modal run modal_train.py --exp-id 003 --mode infer
        modal run modal_train.py --exp-id all    # run full sweep
    """
    if exp_id == "all":
        for eid in SWEEP_1:
            print(f"\n{'='*60}")
            print(f"Starting experiment {eid}")
            print(f"{'='*60}\n")
            train.remote(eid, dataset_name=dataset_name)
    elif mode == "train":
        train.remote(exp_id, dataset_name=dataset_name)
    elif mode == "infer":
        infer.remote(exp_id)
    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'infer'.")
```

### Quick Start

```bash
# 1. Install Modal CLI
pip install modal
modal setup    # one-time auth

# 2. Create secrets (one-time)
modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
modal secret create wandb-secret WANDB_API_KEY=xxxxx

# 3. Run baseline experiment (rank=16, lr=1e-4)
modal run modal_train.py --exp-id 003

# 4. Run full 8-experiment sweep
modal run modal_train.py --exp-id all

# 5. Generate comparison images
modal run modal_train.py --exp-id 003 --mode infer
```

### Custom Dataset

To use your own images instead of `lambdalabs/naruto-blip-captions`:

1. Upload to a HuggingFace dataset, or
2. Upload to Modal Volume:

```bash
# Upload local images to Modal Volume
modal volume put sdxl-lora-experiments ./my-images /data/my-images

# Then pass the data dir path in the training script
# (requires modifying the train function to use --train_data_dir instead of --dataset_name)
```

## Results Tracking

```
# results.tsv
exp_id	rank	alpha	lr	steps	val_loss	fid	clip_score	vram_gb	status	notes
001	8	8	1e-4	1000	-	-	-	-	pending	rank=8 baseline
002	8	8	5e-5	1000	-	-	-	-	pending	rank=8 lower lr
003	16	16	1e-4	1000	-	-	-	-	pending	rank=16 baseline
004	16	16	5e-5	1000	-	-	-	-	pending	rank=16 lower lr
005	32	32	1e-4	1000	-	-	-	-	pending	rank=32 baseline
006	32	32	5e-5	1000	-	-	-	-	pending	rank=32 lower lr
007	64	64	1e-4	1000	-	-	-	-	pending	rank=64 baseline
008	64	64	5e-5	1000	-	-	-	-	pending	rank=64 lower lr
```

## Evaluation Strategy

**Quantitative (per run):**
- Training loss curve (via W&B)
- Validation loss at checkpoints

**Qualitative (compare top runs):**
- Same 4-8 validation prompts rendered at same seed
- Visual side-by-side comparison
- Check for: style adherence, prompt following, artifact-free generation, diversity

**Keep/discard rule:**
- Keep if: visually coherent, matches training domain, no artifacts
- Discard if: mode collapse, heavy artifacts, no visible learning
- Top 2 from Sweep 1 advance to Sweep 2

## Cost Estimate

- A100-80GB on Modal: ~$3.74/hr (on-demand)
- ~15-30 min per 1000-step run (depends on batch size)
- 8-run sweep: ~$15-30 total
- With Sweep 2 (6 more runs): ~$30-50 total
