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
import shutil
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
EXPECTED_EVAL_PROMPTS = NUM_TRIGGER_PROMPTS + NUM_NEG_PROMPTS
NEG_WARN_THRESHOLD = 0.45
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

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
        "git clone --depth 1 --branch v0.31.0 https://github.com/huggingface/diffusers.git /diffusers_repo"
    )
    .add_local_file("score.py", "/root/score.py")
)

app = modal.App("sdxl-autoresearch", image=image)

# ---------------------------------------------------------------------------
# Fixed model config
# ---------------------------------------------------------------------------

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_NAME = "madebyollin/sdxl-vae-fp16-fix"
RESOLUTION = 1024
SEED = 42


SCREEN_STEPS = 50  # Steps for cheap screening tier


def build_caption_prefix(config: dict) -> str:
    trigger = config.get("trigger_word", "cybrn")
    template = config.get("caption_template", "a painting in the style of {trigger}, ")
    return template.replace("{trigger}", trigger)


def materialize_training_data(train_data_dir: str, config: dict, dataset_name: str = "") -> str:
    """Build a local train_data_dir so caption overrides apply per experiment."""
    import io

    from PIL import Image

    target_dir = Path(train_data_dir)
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    image_filenames = []

    if dataset_name:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split="train")
        if len(dataset) == 0:
            raise ValueError(f"Dataset '{dataset_name}' is empty")
        if "image" not in dataset.column_names:
            raise ValueError(
                f"Dataset '{dataset_name}' must expose an 'image' column, found: {dataset.column_names}"
            )

        for idx, example in enumerate(dataset):
            image_obj = example["image"]
            if isinstance(image_obj, Image.Image):
                image = image_obj.convert("RGB")
            elif isinstance(image_obj, dict) and image_obj.get("bytes") is not None:
                image = Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
            elif isinstance(image_obj, (str, os.PathLike)):
                image = Image.open(image_obj).convert("RGB")
            else:
                raise ValueError(f"Unsupported image payload type for dataset '{dataset_name}': {type(image_obj)!r}")

            filename = f"{idx:05d}.png"
            image.save(target_dir / filename)
            image_filenames.append(filename)
    else:
        source_dir = Path(f"{VOL_DIR}/autoresearch/train_data")
        if not source_dir.exists():
            raise ValueError(
                f"Training data directory not found: {source_dir}. Run 'modal run prepare.py --images <dir>' first."
            )

        # Copy images into per-experiment dir (parallel-safe, each experiment gets its own copy)
        for src_path in sorted(source_dir.iterdir()):
            if (
                not src_path.is_file()
                or src_path.name == "metadata.jsonl"
                or src_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
            ):
                continue
            dest = target_dir / src_path.name
            if not dest.exists():
                shutil.copy2(src_path, dest)
            image_filenames.append(src_path.name)

    if not image_filenames:
        raise ValueError("No training images found to materialize")

    caption_prefix = build_caption_prefix(config)
    metadata_path = target_dir / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for filename in image_filenames:
            f.write(json.dumps({"file_name": filename, "text": caption_prefix}) + "\n")

    return str(target_dir)


def merge_experiment_configs(base_config: dict, experiments: list[dict], default_tag_prefix: str) -> tuple[list[dict], list[str]]:
    """Merge batch overrides and reject duplicate tags before launching work."""
    configs = []
    tags = []
    seen_tags = set()

    for idx, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise ValueError(f"Experiment #{idx + 1} must be a YAML mapping, got {type(exp).__name__}")

        tag = exp.get("tag", f"{default_tag_prefix}_{idx}")
        if not tag:
            raise ValueError(f"Experiment #{idx + 1} has an empty tag")
        if tag in seen_tags:
            raise ValueError(f"Duplicate experiment tag '{tag}' in batch.yaml; tags must be unique")

        merged = {**base_config, **{k: v for k, v in exp.items() if k != "tag"}}
        configs.append(merged)
        tags.append(tag)
        seen_tags.add(tag)

    return configs, tags


@app.function(
    gpu="A100-80GB",
    volumes={VOL_DIR: volume},
    timeout=15 * MINUTES,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def screen_experiment(config: dict, exp_tag: str = "screen", dataset_name: str = ""):
    """Cheap screening tier: train 50 steps, report loss trajectory only.

    No image generation, no CLIP scoring. ~5 min per run.
    Used to filter out bad configs before expensive full evaluation.
    """
    import re
    import torch
    from accelerate.utils import write_basic_config

    output_dir = f"{VOL_DIR}/autoresearch/runs/{exp_tag}"
    os.makedirs(output_dir, exist_ok=True)

    write_basic_config(mixed_precision="bf16")
    t_start = time.time()

    try:
        train_data_dir = materialize_training_data(f"{output_dir}/train_data", config, dataset_name)
    except Exception as e:
        return {
            "status": "crash",
            "error": f"train_data_failed: {e}",
            "training_seconds": time.time() - t_start,
            "exp_tag": exp_tag,
            "config": config,
        }

    # Override steps to SCREEN_STEPS, disable validation (no image gen)
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
        f"--lr_warmup_steps={min(config.get('lr_warmup_steps', 100), SCREEN_STEPS // 2)}",
        f"--max_train_steps={SCREEN_STEPS}",
        f"--checkpointing_steps={SCREEN_STEPS}",
        f"--seed={SEED}",
        f"--output_dir={output_dir}",
        f"--rank={config.get('rank', 16)}",
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        "--report_to=tensorboard",
    ]

    cmd.extend([f"--train_data_dir={train_data_dir}", "--caption_column=text"])

    print(f"=== [{exp_tag}] SCREENING ({SCREEN_STEPS} steps) ===", flush=True)
    print(
        f"Config: rank={config.get('rank')}, lr={config['lr']}, caption='{build_caption_prefix(config)}'"
    )

    # Capture output to parse loss values
    losses = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with process.stdout as pipe:
        for line in iter(pipe.readline, b""):
            text = line.decode()
            print(text, end="")
            # Parse step_loss from progress bar output
            match = re.search(r"step_loss=([\d.e+-]+)", text)
            if match:
                losses.append(float(match.group(1)))

    exit_code = process.wait()
    training_seconds = time.time() - t_start

    if exit_code != 0 or len(losses) < 5:
        return {
            "status": "crash",
            "error": "screen_failed",
            "training_seconds": training_seconds,
            "exp_tag": exp_tag,
            "config": config,
        }

    # Compute loss trajectory metrics
    import numpy as np
    losses_arr = np.array(losses)
    first_half = losses_arr[: len(losses_arr) // 2]
    second_half = losses_arr[len(losses_arr) // 2 :]

    avg_loss = float(np.mean(losses_arr))
    loss_slope = float(np.mean(second_half) - np.mean(first_half))  # negative = improving
    final_loss = float(np.mean(losses_arr[-5:]))

    print("---")
    print(f"exp_tag:            {exp_tag}")
    print(f"avg_loss:           {avg_loss:.6f}")
    print(f"final_loss:         {final_loss:.6f}")
    print(f"loss_slope:         {loss_slope:.6f}")
    print(f"num_loss_samples:   {len(losses)}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print("---")

    return {
        "status": "ok",
        "exp_tag": exp_tag,
        "config": config,
        "avg_loss": avg_loss,
        "final_loss": final_loss,
        "loss_slope": loss_slope,
        "num_loss_samples": len(losses),
        "training_seconds": training_seconds,
    }


@app.function(
    gpu="A100-80GB",
    volumes={VOL_DIR: volume},
    timeout=50 * MINUTES,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("openrouter-secret", required_keys=["OPENROUTER_API_KEY"]),
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
        aggregate_multi_scores,
        embed_image,
        hpsv2_score,
        pickscore,
        score_against_centroid,
        score_nearest_neighbor,
        vlm_judge,
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
    caption_prefix = build_caption_prefix(config)

    try:
        train_data_dir = materialize_training_data(f"{output_dir}/train_data", config, dataset_name)
    except Exception as e:
        return {"status": "crash", "error": f"train_data_failed: {e}", "exp_tag": exp_tag, "config": config}

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

    # Materialized local data makes caption overrides apply consistently.
    cmd.extend([
        f"--train_data_dir={train_data_dir}",
        "--caption_column=text",
    ])

    print(
        f"Config: rank={config.get('rank')}, lr={config['lr']}, steps={config.get('max_train_steps')}, "
        f"caption='{caption_prefix}'"
    )

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

    # --- MULTI-METRIC SCORING ---
    print(f"\n=== [{exp_tag}] SCORING (CLIP + PickScore + HPSv2) ===", flush=True)

    centroid_path = f"{ref_dir}/ref_centroid.npy"
    embeddings_path = f"{ref_dir}/ref_embeddings.npy"

    if not os.path.exists(centroid_path):
        return {"status": "crash", "error": "no_reference_embeddings", "exp_tag": exp_tag, "config": config}

    centroid = np.load(centroid_path)
    ref_embeddings = np.load(embeddings_path)

    trigger_centroid_sims = []
    trigger_nn_sims = []
    neg_sims = []
    trigger_pickscores = []
    trigger_hpsv2 = []
    trigger_vlm = []

    for pi, seed, img_path in image_paths:
        img_p = Path(img_path)
        prompt = prompts[pi]

        # CLIP scoring
        emb = embed_image(img_p)
        c_sim = score_against_centroid(emb, centroid)
        nn_sim = score_nearest_neighbor(emb, ref_embeddings)

        if pi < NUM_TRIGGER_PROMPTS:
            trigger_centroid_sims.append(c_sim)
            trigger_nn_sims.append(nn_sim)

            # PickScore + HPSv2 (only for trigger images, not negative control)
            try:
                ps = pickscore(img_p, prompt)
                trigger_pickscores.append(ps)
            except Exception as e:
                print(f"  PickScore failed: {e}")
                trigger_pickscores.append(0.0)

            try:
                hs = hpsv2_score(img_p, prompt)
                trigger_hpsv2.append(hs)
            except Exception as e:
                print(f"  HPSv2 failed: {e}")
                trigger_hpsv2.append(0.0)

            # VLM judge (only if API key available, sample 1 per prompt to control cost)
            if seed == EVAL_SEEDS[0]:
                vj = vlm_judge(img_p)
                if vj["vlm_avg"] > 0:
                    trigger_vlm.append(vj["vlm_avg"])
        else:
            neg_sims.append(c_sim)

        print(f"  [{pi}:{seed}] clip={c_sim:.3f}", end="")
        if pi < NUM_TRIGGER_PROMPTS:
            print(f"  pick={trigger_pickscores[-1]:.3f}  hps={trigger_hpsv2[-1]:.3f}", end="")
        print()

    if not trigger_centroid_sims:
        return {"status": "crash", "error": "no_scores", "exp_tag": exp_tag, "config": config}

    scores = aggregate_multi_scores(
        trigger_centroid_sims, trigger_nn_sims,
        neg_sims if neg_sims else [0.0],
        trigger_pickscores,
        trigger_hpsv2,
        trigger_vlm,
        num_prompts=NUM_TRIGGER_PROMPTS,
        seeds_per_prompt=len(EVAL_SEEDS),
    )

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # --- OUTPUT ---
    prompt_scores_str = ", ".join(f"{s:.2f}" for s in scores["prompt_scores"])
    print("---")
    print(f"exp_tag:            {exp_tag}")
    print(f"composite_score:    {scores['composite_score']:.6f}")
    print(f"clip_sim_centroid:  {scores['clip_sim_centroid']:.6f}")
    print(f"clip_sim_nn:        {scores['clip_sim_nn']:.6f}")
    print(f"pickscore_avg:      {scores['pickscore_avg']:.6f}")
    print(f"hpsv2_avg:          {scores['hpsv2_avg']:.6f}")
    print(f"vlm_avg:            {scores['vlm_avg']:.6f}")
    print(f"prompt_scores:      {prompt_scores_str}")
    print(f"score_stddev:       {scores['score_stddev']:.6f}")
    print(f"neg_control:        {scores['neg_control']:.6f}")
    print(f"peak_vram_gb:       {peak_gb:.1f}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"steps_completed:    {config.get('max_train_steps', 0)}")
    print(f"eval_seconds:       {eval_seconds:.1f}")
    print("---")

    if scores["neg_control"] > NEG_WARN_THRESHOLD:
        print(f"WARNING: neg_control ({scores['neg_control']:.3f}) > {NEG_WARN_THRESHOLD} — possible overfitting")

    return {
        "status": "ok",
        "exp_tag": exp_tag,
        "config": config,
        **scores,
        "peak_vram_gb": peak_gb,
        "training_seconds": training_seconds,
        "eval_seconds": eval_seconds,
        "steps_completed": config.get("max_train_steps", 0),
    }


def load_eval_prompts(project_dir: Path) -> list[str]:
    """Load eval prompts from file."""
    prompts_path = project_dir / "eval_prompts.txt"
    prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]

    if len(prompts) != EXPECTED_EVAL_PROMPTS:
        raise ValueError(
            f"{prompts_path} must contain exactly {EXPECTED_EVAL_PROMPTS} prompts "
            f"({NUM_TRIGGER_PROMPTS} trigger + {NUM_NEG_PROMPTS} negative control), found {len(prompts)}"
        )

    return prompts


@app.local_entrypoint()
def main(dry_run: bool = False, batch: bool = False, screen: bool = False, dataset_name: str = ""):
    """Entry point.

    Single mode:  reads config.yaml, runs one experiment.
    Batch mode:   reads batch.yaml, runs N experiments in parallel on separate GPUs.
    Screen mode:  reads batch.yaml, runs cheap 50-step screening on all configs in parallel.
                  Reports loss trajectory only — no image generation, no CLIP scoring. ~5 min.
    """
    project_dir = Path(__file__).parent
    eval_prompts = load_eval_prompts(project_dir)

    if screen:
        # --- SCREEN MODE: cheap 50-step loss-only screening ---
        batch_path = project_dir / "batch.yaml"
        if not batch_path.exists():
            print("ERROR: batch.yaml not found for screening")
            sys.exit(1)

        batch_config = yaml.safe_load(batch_path.read_text())
        base_config = yaml.safe_load((project_dir / "config.yaml").read_text())
        experiments = batch_config.get("experiments", [])

        configs, tags = merge_experiment_configs(base_config, experiments, "screen")

        if dry_run:
            print(f"DRY RUN — screening {len(configs)} configs ({SCREEN_STEPS} steps each):\n")
            for tag, config in zip(tags, configs):
                print(f"  [{tag}] rank={config.get('rank')}, lr={config.get('lr')}")
            sys.exit(0)

        print(f"Screening {len(configs)} configs ({SCREEN_STEPS} steps each, parallel)...\n")

        results = list(screen_experiment.map(
            configs,
            tags,
            [dataset_name] * len(configs),
        ))

        # --- SCREEN RESULTS ---
        print(f"\n{'='*70}")
        print(f"SCREEN RESULTS — {len(results)} configs ({SCREEN_STEPS} steps each)")
        print(f"{'='*70}\n")

        ok_results = []
        for r in results:
            tag = r.get("exp_tag", "?")
            if r["status"] == "ok":
                ok_results.append(r)
                cfg = r.get("config", {})
                print(f"  [{tag}] final_loss={r['final_loss']:.4f}  slope={r['loss_slope']:.4f}  "
                      f"rank={cfg.get('rank')}  lr={cfg.get('lr')}  "
                      f"time={r['training_seconds']:.0f}s")
            else:
                print(f"  [{tag}] FAILED: {r.get('error', 'unknown')}")

        if ok_results:
            # Best = lowest final loss (most negative slope is also good)
            best = min(ok_results, key=lambda r: r["final_loss"])
            worst = max(ok_results, key=lambda r: r["final_loss"])
            print(f"\n{'─'*70}")
            print(f"BEST:  [{best['exp_tag']}] final_loss={best['final_loss']:.4f}  slope={best['loss_slope']:.4f}")
            print(f"WORST: [{worst['exp_tag']}] final_loss={worst['final_loss']:.4f}  slope={worst['loss_slope']:.4f}")
            print(f"{'─'*70}")

            # Rank all by final_loss for the LLM to parse
            ranked = sorted(ok_results, key=lambda r: r["final_loss"])
            print(f"\nscreen_ranking:")
            for i, r in enumerate(ranked):
                cfg = r.get("config", {})
                print(f"  {i+1}. [{r['exp_tag']}] final_loss={r['final_loss']:.4f} config={json.dumps({k: cfg[k] for k in ['rank', 'lr', 'max_train_steps'] if k in cfg})}")

    elif batch:
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

        configs, tags = merge_experiment_configs(base_config, experiments, "exp")

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
                print(f"  [{tag}] composite={r['composite_score']:.4f}  clip={r['clip_sim_centroid']:.4f}  "
                      f"pick={r['pickscore_avg']:.4f}  hps={r['hpsv2_avg']:.4f}  vlm={r['vlm_avg']:.4f}  "
                      f"neg={r['neg_control']:.4f}  rank={cfg.get('rank')}  lr={cfg.get('lr')}  "
                      f"train={r['training_seconds']:.0f}s")
            else:
                print(f"  [{tag}] FAILED: {r.get('error', 'unknown')}")

        if ok_results:
            # Rank by composite score, keep top 3
            ranked = sorted(ok_results, key=lambda r: r["composite_score"], reverse=True)
            top_n = min(3, len(ranked))

            print(f"\n{'─'*70}")
            print(f"TOP {top_n} (ranked by composite score):")
            print(f"{'─'*70}")
            for i, r in enumerate(ranked[:top_n]):
                cfg = r.get("config", {})
                print(f"  #{i+1} [{r['exp_tag']}] composite={r['composite_score']:.4f}  "
                      f"clip={r['clip_sim_centroid']:.4f}  pick={r['pickscore_avg']:.4f}  "
                      f"hps={r['hpsv2_avg']:.4f}  rank={cfg.get('rank')}  lr={cfg.get('lr')}")

            best = ranked[0]
            cfg = best.get("config", {})
            print(f"\nbest_tag: {best['exp_tag']}")
            print(f"best_composite: {best['composite_score']:.6f}")
            print(f"best_clip_centroid: {best['clip_sim_centroid']:.6f}")
            print(f"best_pickscore: {best['pickscore_avg']:.6f}")
            print(f"best_hpsv2: {best['hpsv2_avg']:.6f}")
            print(f"best_config: {json.dumps(cfg)}")

            if top_n >= 2:
                print(f"runner_up_tag: {ranked[1]['exp_tag']}")
                print(f"runner_up_composite: {ranked[1]['composite_score']:.6f}")

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
