"""
Fuse renaissance paintings with cyberpunk aesthetic using SDXL img2img.

Reads renaissance paintings from data/renaissance-originals/, runs them through
SDXL img2img with cyberpunk prompts at varying strengths to blend the two aesthetics.

Usage:
    # Generate fused dataset
    modal run scripts/fuse_dataset.py

    # Single strength for faster test
    modal run scripts/fuse_dataset.py --strengths 0.55

    # Custom output
    modal run scripts/fuse_dataset.py --output-dir dataset/fused-v2
"""
import io
import os
from pathlib import Path

import modal

MINUTES = 60

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
        "safetensors==0.4.4",
        "huggingface-hub==0.36.0",
        "Pillow>=10.0.0",
        "numpy<2",
    )
    .env({"HF_HUB_CACHE": "/cache", "HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("sdxl-fuse-dataset", image=image)

# Cyberpunk fusion prompts — applied over the renaissance composition
FUSION_PROMPTS = [
    "cybernetic implants, glowing neon circuitry under skin, holographic elements, oil painting with neon accents",
    "chrome and metallic surfaces, LED highlights, data streams, renaissance oil painting fused with cyberpunk technology",
    "neon-lit environment, circuit board patterns, holographic overlays, classical painting reimagined as cyber renaissance",
    "bioluminescent accents, fiber optic details, plasma glow, baroque lighting meets futuristic technology, oil painting texture",
]

NEGATIVE_PROMPT = (
    "ugly, blurry, low quality, deformed, disfigured, "
    "cartoon, anime, 3d render, photograph, modern clothing, "
    "plain background, simple, flat colors, watermark, text"
)


@app.function(
    gpu="A100-80GB",
    volumes={VOL_DIR: volume},
    timeout=60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def fuse_batch(
    painting_bytes_list: list[bytes],
    painting_names: list[str],
    fusion_prompts: list[str],
    strengths: list[float],
    output_dir: str,
    negative_prompt: str,
):
    """Fuse a batch of renaissance paintings with cyberpunk prompts via img2img."""
    import torch
    from diffusers import AutoPipelineForImage2Image
    from PIL import Image

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")

    os.makedirs(output_dir, exist_ok=True)
    idx = 0

    for img_bytes, name in zip(painting_bytes_list, painting_names):
        source = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        source = source.resize((1024, 1024), Image.LANCZOS)

        for pi, prompt_text in enumerate(fusion_prompts):
            for strength in strengths:
                seed = 42 + idx
                generator = torch.Generator(device="cuda").manual_seed(seed)

                result = pipe(
                    prompt=prompt_text,
                    negative_prompt=negative_prompt,
                    image=source,
                    strength=strength,
                    num_inference_steps=40,
                    guidance_scale=7.5,
                    generator=generator,
                ).images[0]

                img_path = f"{output_dir}/{name}_p{pi}_s{strength:.2f}.png"
                result.save(img_path)

                caption = f"a cyber renaissance painting, {prompt_text}"
                txt_path = img_path.replace(".png", ".txt")
                with open(txt_path, "w") as f:
                    f.write(caption)

                idx += 1

        print(f"  {name}: {len(fusion_prompts) * len(strengths)} fusions done")

    return idx


@app.local_entrypoint()
def main(
    images_dir: str = "data/renaissance-originals",
    output_dir: str = "dataset/cyber-renaissance-fused",
    strengths: str = "0.45,0.55,0.65",
    max_paintings: int = 150,
    batch_size: int = 5,
):
    strength_list = [float(s) for s in strengths.split(",")]
    remote_dir = f"{VOL_DIR}/{output_dir}"
    images_path = Path(images_dir)

    # Load local paintings
    exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([f for f in images_path.iterdir() if f.suffix.lower() in exts])[:max_paintings]

    if not image_files:
        print(f"No images found in {images_path}")
        return

    total_fusions = len(image_files) * len(FUSION_PROMPTS) * len(strength_list)
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    print(f"Source: {len(image_files)} paintings from {images_path}")
    print(f"Fusions: {len(FUSION_PROMPTS)} prompts × {len(strength_list)} strengths = {len(FUSION_PROMPTS) * len(strength_list)} per painting")
    print(f"Total: {total_fusions} fused images")
    print(f"Batches: {num_batches} (batch_size={batch_size}, each batch = 1 GPU)")
    print(f"Strengths: {strength_list}\n")

    # Build batches
    all_bytes = []
    all_names = []
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i : i + batch_size]
        all_bytes.append([f.read_bytes() for f in batch_files])
        all_names.append([f.stem for f in batch_files])

    print(f"Launching {len(all_bytes)} batches in parallel on separate GPUs...")

    results = list(fuse_batch.map(
        all_bytes,
        all_names,
        [FUSION_PROMPTS] * len(all_bytes),
        [strength_list] * len(all_bytes),
        [remote_dir] * len(all_bytes),
        [NEGATIVE_PROMPT] * len(all_bytes),
    ))

    total = sum(results)
    print(f"\nDone! {total} fused images generated.")
    print(f"Download: uv run modal volume get sdxl-lora-experiments {output_dir} data/fused")
