"""
Autonomous SDXL LoRA training pipeline on Modal.
Supports single runs and parallel batch experiments.

Ported from autoresearch-lora (FLUX/mflux/local → SDXL/Diffusers/Modal).

Usage:
    # Single experiment (reads config.yaml)
    modal run train.py

    # Parallel batch (reads batch.yaml — multiple configs at once)
    modal run train.py --batch

    # Dry run
    modal run train.py --dry-run
    modal run train.py --batch --dry-run
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINUTES = 60
EVAL_SEEDS = [42, 137, 256, 999]
EVAL_STEPS = 30
NUM_TRIGGER_PROMPTS = 5  # Prompts 1-5 (with trigger)
NUM_NEG_PROMPTS = 1  # Prompt 6 (negative control, no LoRA influence)
NEG_WARN_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("sdxl-lora-experiments", create_if_missing=True)
VOL_DIR = "/vol"

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
        "git clone --depth 1 https://github.com/huggingface/diffusers.git /diffusers_repo"
    )
)

app = modal.App(
    "sdxl-autoresearch",
    image=image,
    mounts=[modal.Mount.from_local_file("score.py", remote_path="/root/score.py")],
)

# ---------------------------------------------------------------------------
# Fixed model config
# ---------------------------------------------------------------------------

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_NAME = "madebyollin/sdxl-vae-fp16-fix"
RESOLUTION = 1024
SEED = 42


@app.function(
    gpu="A100-80GB",
    volumes={VOL_DIR: volume},
    timeout=60 * MINUTES,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run_experiment(config: dict, eval_prompts: list[str], exp_tag: str = "current", dataset_name: str = ""):
    """Full pipeline: train → generate → CLIP score. Returns metrics dict.

    Each experiment writes to its own directory (exp_tag) to avoid
    conflicts when running multiple experiments in parallel.
    """
    import numpy as np
    import torch
    from accelerate.utils import write_basic_config
    from diffusers import DiffusionPipeline

    from score import (
        aggregate_scores,
        embed_image,
        score_against_centroid,
        score_nearest_neighbor,
    )

    output_dir = f"{VOL_DIR}/autoresearch/runs/{exp_tag}"
    eval_dir = f"{VOL_DIR}/autoresearch/runs/{exp_tag}/eval_images"
    ref_dir = f"{VOL_DIR}/autoresearch/reference"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # --- TRAIN ---
    print(f"=== [{exp_tag}] TRAINING ===", flush=True)
    write_basic_config(mixed_precision="bf16")
    t_train_start = time.time()

    trigger = config.get("trigger_word", "cybrn")
    caption_prefix = config.get("caption_template", "a painting in the style of {trigger}, ").replace("{trigger}", trigger)

    cmd = [
        "accelerate", "launch",
        "/diffusers_repo/examples/text_to_image/train_text_to_image_lora_sdxl.py",
        f"--pretrained_model_name_or_path={MODEL_NAME}",
        f"--pretrained_vae_model_name_or_path={VAE_NAME}",
        f"--resolution={RESOLUTION}",
        f"--train_batch_size={config.get('train_batch_size', 1)}",
        f"--gradient_accumulation_steps={config.get('gradient_accumulation_steps', 4)}",
        f"--learning_rate={config['lr']}",
        f"--lr_scheduler={config.get('lr_scheduler', 'cosine')}",
        f"--lr_warmup_steps={config.get('lr_warmup_steps', 100)}",
        f"--max_train_steps={config.get('max_train_steps', 1000)}",
        f"--checkpointing_steps={config.get('max_train_steps', 1000)}",
        f"--seed={SEED}",
        f"--output_dir={output_dir}",
        f"--rank={config.get('rank', 16)}",
        # Note: alpha is hardcoded to equal rank in the Diffusers SDXL LoRA script.
        # To use alpha != rank, you would need to fork the training script.
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        f"--validation_prompt=a painting in the style of {trigger}, a renaissance portrait with neon accents",
        "--validation_epochs=1",
        "--report_to=tensorboard",
    ]

    # Dataset source: HF dataset or local data dir on volume
    if dataset_name:
        cmd.extend([
            f"--dataset_name={dataset_name}",
            "--caption_column=text",
        ])
    else:
        train_data_dir = f"{VOL_DIR}/autoresearch/train_data"
        cmd.extend([
            f"--train_data_dir={train_data_dir}",
            "--caption_column=text",
        ])

    print(f"Config: rank={config.get('rank')}, lr={config['lr']}, steps={config.get('max_train_steps')}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            print(line.decode(), end="")

    exit_code = process.wait()
    training_seconds = time.time() - t_train_start

    if exit_code != 0:
        print(f"[{exp_tag}] TRAINING FAILED (exit code {exit_code})")
        return {"status": "crash", "error": "training_failed", "training_seconds": training_seconds, "exp_tag": exp_tag, "config": config}

    print(f"[{exp_tag}] Training complete ({training_seconds:.1f}s)")

    # --- GENERATE EVAL IMAGES ---
    print(f"\n=== [{exp_tag}] GENERATING EVAL IMAGES ===", flush=True)
    t_eval_start = time.time()

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, use_safetensors=True,
    ).to("cuda")
    pipe.load_lora_weights(output_dir)

    prompts = [p.replace("{trigger}", trigger) for p in eval_prompts]
    image_paths = []

    # Generate LoRA images (trigger prompts)
    for pi, prompt in enumerate(prompts[:NUM_TRIGGER_PROMPTS]):
        for seed in EVAL_SEEDS:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            img = pipe(
                prompt,
                num_inference_steps=EVAL_STEPS,
                guidance_scale=config.get("guidance", 7.5),
                width=RESOLUTION, height=RESOLUTION,
                generator=generator,
            ).images[0]

            img_path = f"{eval_dir}/p{pi}_s{seed}.png"
            img.save(img_path)
            image_paths.append((pi, seed, img_path))

    # Unload LoRA for negative control images (saves ~6.5GB VRAM vs loading a second pipeline)
    pipe.unload_lora_weights()

    for pi, prompt in enumerate(prompts[NUM_TRIGGER_PROMPTS:], start=NUM_TRIGGER_PROMPTS):
        for seed in EVAL_SEEDS:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            img = pipe(
                prompt,
                num_inference_steps=EVAL_STEPS,
                guidance_scale=config.get("guidance", 7.5),
                width=RESOLUTION, height=RESOLUTION,
                generator=generator,
            ).images[0]

            img_path = f"{eval_dir}/p{pi}_s{seed}.png"
            img.save(img_path)
            image_paths.append((pi, seed, img_path))

    eval_seconds = time.time() - t_eval_start
    print(f"[{exp_tag}] Generated {len(image_paths)} images ({eval_seconds:.1f}s)")

    # Free VRAM
    del pipe
    torch.cuda.empty_cache()

    # --- CLIP SCORING ---
    print(f"\n=== [{exp_tag}] CLIP SCORING ===", flush=True)

    centroid_path = f"{ref_dir}/ref_centroid.npy"
    embeddings_path = f"{ref_dir}/ref_embeddings.npy"

    if not os.path.exists(centroid_path):
        return {"status": "crash", "error": "no_reference_embeddings", "exp_tag": exp_tag, "config": config}

    centroid = np.load(centroid_path)
    ref_embeddings = np.load(embeddings_path)

    trigger_centroid_sims = []
    trigger_nn_sims = []
    neg_sims = []

    for pi, seed, img_path in image_paths:
        emb = embed_image(Path(img_path))
        c_sim = score_against_centroid(emb, centroid)
        nn_sim = score_nearest_neighbor(emb, ref_embeddings)

        if pi < NUM_TRIGGER_PROMPTS:
            trigger_centroid_sims.append(c_sim)
            trigger_nn_sims.append(nn_sim)
        else:
            neg_sims.append(c_sim)

    if not trigger_centroid_sims:
        return {"status": "crash", "error": "no_scores", "exp_tag": exp_tag, "config": config}

    scores = aggregate_scores(
        trigger_centroid_sims, trigger_nn_sims,
        neg_sims if neg_sims else [0.0],
        num_prompts=NUM_TRIGGER_PROMPTS,
        seeds_per_prompt=len(EVAL_SEEDS),
    )

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --- OUTPUT ---
    prompt_scores_str = ", ".join(f"{s:.2f}" for s in scores["prompt_scores"])
    print("---")
    print(f"exp_tag:            {exp_tag}")
    print(f"clip_sim_centroid:  {scores['clip_sim_centroid']:.6f}")
    print(f"clip_sim_nn:        {scores['clip_sim_nn']:.6f}")
    print(f"prompt_scores:      {prompt_scores_str}")
    print(f"score_stddev:       {scores['score_stddev']:.6f}")
    print(f"neg_control:        {scores['neg_control']:.6f}")
    print(f"peak_vram_mb:       {peak_mb:.1f}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"steps_completed:    {config.get('max_train_steps', 0)}")
    print(f"eval_seconds:       {eval_seconds:.1f}")
    print("---")

    if scores["neg_control"] > NEG_WARN_THRESHOLD:
        print(f"WARNING: neg_control ({scores['neg_control']:.3f}) > {NEG_WARN_THRESHOLD} — possible overfitting")

    # Note: volume.commit() is NOT called here. When running parallel experiments
    # via .map(), concurrent commits can overwrite each other's snapshots.
    # Modal auto-persists writes to the volume when the function completes.

    return {
        "status": "ok",
        "exp_tag": exp_tag,
        "config": config,
        **scores,
        "peak_vram_mb": peak_mb,
        "training_seconds": training_seconds,
        "eval_seconds": eval_seconds,
        "steps_completed": config.get("max_train_steps", 0),
    }


def load_eval_prompts(project_dir: Path) -> list[str]:
    """Load eval prompts from file."""
    prompts_path = project_dir / "eval_prompts.txt"
    return [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]


@app.local_entrypoint()
def main(dry_run: bool = False, batch: bool = False, dataset_name: str = ""):
    """Entry point.

    Single mode: reads config.yaml, runs one experiment.
    Batch mode:  reads batch.yaml, runs N experiments in parallel on separate GPUs.
    """
    project_dir = Path(__file__).parent
    eval_prompts = load_eval_prompts(project_dir)

    if batch:
        # --- BATCH MODE: parallel experiments ---
        batch_path = project_dir / "batch.yaml"
        if not batch_path.exists():
            print("ERROR: batch.yaml not found. Create it with a list of experiment configs.")
            print("Example batch.yaml:")
            print("  experiments:")
            print("    - tag: rank8")
            print("      rank: 8")
            print("    - tag: rank32")
            print("      rank: 32")
            sys.exit(1)

        batch_config = yaml.safe_load(batch_path.read_text())
        base_config = yaml.safe_load((project_dir / "config.yaml").read_text())
        experiments = batch_config.get("experiments", [])

        if not experiments:
            print("ERROR: batch.yaml has no experiments")
            sys.exit(1)

        # Merge each experiment's overrides with the base config
        configs = []
        tags = []
        for exp in experiments:
            tag = exp.get("tag", f"exp_{len(configs)}")
            merged = {**base_config, **{k: v for k, v in exp.items() if k != "tag"}}
            configs.append(merged)
            tags.append(tag)

        if dry_run:
            print(f"DRY RUN — {len(configs)} parallel experiments:\n")
            for tag, config in zip(tags, configs):
                print(f"  [{tag}] rank={config.get('rank')}, lr={config.get('lr')}, steps={config.get('max_train_steps')}")
            sys.exit(0)

        print(f"Launching {len(configs)} experiments in parallel...")
        for tag, config in zip(tags, configs):
            print(f"  [{tag}] rank={config.get('rank')}, lr={config.get('lr')}, steps={config.get('max_train_steps')}")

        # Fan out to parallel GPUs using Modal's .map()
        results = list(run_experiment.map(
            configs,
            [eval_prompts] * len(configs),
            tags,
            [dataset_name] * len(configs),
        ))

        # --- RESULTS SUMMARY ---
        print(f"\n{'='*70}")
        print(f"BATCH RESULTS — {len(results)} experiments")
        print(f"{'='*70}\n")

        ok_results = []
        for r in results:
            tag = r.get("exp_tag", "?")
            if r["status"] == "ok":
                ok_results.append(r)
                cfg = r.get("config", {})
                print(f"  [{tag}] clip_centroid={r['clip_sim_centroid']:.4f}  nn={r['clip_sim_nn']:.4f}  neg={r['neg_control']:.4f}  "
                      f"rank={cfg.get('rank')}  lr={cfg.get('lr')}  steps={cfg.get('max_train_steps')}  "
                      f"train={r['training_seconds']:.0f}s")
            else:
                print(f"  [{tag}] FAILED: {r.get('error', 'unknown')}")

        if ok_results:
            best = max(ok_results, key=lambda r: r["clip_sim_centroid"])
            print(f"\n{'─'*70}")
            print(f"BEST: [{best['exp_tag']}] clip_sim_centroid={best['clip_sim_centroid']:.6f}")
            cfg = best.get("config", {})
            print(f"  rank={cfg.get('rank')}, lr={cfg.get('lr')}, steps={cfg.get('max_train_steps')}")
            print(f"{'─'*70}")

            # Output JSON for LLM to parse
            print(f"\nbest_tag: {best['exp_tag']}")
            print(f"best_clip_centroid: {best['clip_sim_centroid']:.6f}")
            print(f"best_config: {json.dumps(cfg)}")

    else:
        # --- SINGLE MODE: one experiment ---
        config_path = project_dir / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict):
            print("ERROR: config.yaml must be a YAML mapping")
            sys.exit(1)

        if dry_run:
            print("DRY RUN — config:")
            print(yaml.dump(config, default_flow_style=False))
            print(f"Eval prompts ({len(eval_prompts)}):")
            for p in eval_prompts:
                print(f"  {p}")
            sys.exit(0)

        print(f"Running experiment: rank={config.get('rank')}, lr={config.get('lr')}, steps={config.get('max_train_steps')}")
        result = run_experiment.remote(config, eval_prompts, "current", dataset_name)

        if result["status"] == "ok":
            print(f"\nclip_sim_centroid: {result['clip_sim_centroid']:.6f}")
        else:
            print(f"\nEXPERIMENT FAILED: {result.get('error', 'unknown')}")
