"""
SDXL LoRA training on Modal.

Usage:
    # Single experiment
    modal run modal_train.py --exp-id 003

    # Full sweep
    modal run modal_train.py --exp-id all

    # Generate comparison images
    modal run modal_train.py --exp-id 003 --mode infer
"""
import os
import subprocess

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

    if exp_id not in SWEEP_1:
        raise ValueError(f"Unknown experiment ID: {exp_id}. Valid: {list(SWEEP_1.keys())}")

    config = {**BASE_CONFIG, **SWEEP_1[exp_id]}
    exp_output_dir = f"{OUTPUT_DIR}/exp_{exp_id}"
    os.makedirs(exp_output_dir, exist_ok=True)

    write_basic_config(mixed_precision="bf16")

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
        modal run modal_train.py --exp-id all
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
