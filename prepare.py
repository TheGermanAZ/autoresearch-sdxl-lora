"""
One-time setup for autonomous SDXL LoRA training.
Validates reference images, uploads to Modal Volume, computes CLIP embeddings.

Usage:
    # With local images (e.g., from collect_images.py)
    modal run prepare.py --images data/cyber-renaissance

    # With a HuggingFace dataset
    modal run prepare.py --images data/cyber-renaissance --dataset-name your-user/cyber-renaissance
"""
import os
import shutil
import sys
from pathlib import Path

import modal

volume = modal.Volume.from_name("sdxl-lora-experiments", create_if_missing=True)
VOL_DIR = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "Pillow>=10.0.0",
        "numpy<2",
        "pyyaml",
    )
    .env({"HF_HUB_CACHE": "/cache"})
)

app = modal.App("sdxl-autoresearch-prepare", image=image)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_IMAGES = 10
MIN_RESOLUTION = 512


@app.function(
    gpu="A10G",
    volumes={VOL_DIR: volume},
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def compute_embeddings(image_bytes_list: list[bytes], filenames: list[str], caption_prefix: str):
    """Compute CLIP reference embeddings and store on volume."""
    import io

    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    ref_dir = f"{VOL_DIR}/autoresearch/reference"
    train_data_dir = f"{VOL_DIR}/autoresearch/train_data"
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)

    # Save images to volume as training data
    for img_bytes, filename in zip(image_bytes_list, filenames):
        img_path = f"{train_data_dir}/{filename}"
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        # Write caption .txt file alongside each image
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        with open(txt_path, "w") as f:
            f.write(caption_prefix + "\n")

    # Compute CLIP embeddings
    print("Loading CLIP model...")
    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to("cuda")
    model.eval()

    embeddings = []
    for img_bytes, filename in zip(image_bytes_list, filenames):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy().flatten())
        print(f"  Embedded {filename}")

    embeddings_array = np.array(embeddings)
    centroid = np.mean(embeddings_array, axis=0)

    np.save(f"{ref_dir}/ref_centroid.npy", centroid)
    np.save(f"{ref_dir}/ref_embeddings.npy", embeddings_array)

    volume.commit()
    print(f"Saved centroid + {len(embeddings)} embeddings to volume")
    print(f"Saved {len(filenames)} training images to volume")
    return len(embeddings)


@app.local_entrypoint()
def main(images: str, dataset_name: str = ""):
    import yaml

    images_dir = Path(images)
    if not images_dir.is_dir():
        print(f"Error: {images_dir} is not a directory")
        sys.exit(1)

    # Find images
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    ])

    if len(image_files) == 0:
        print("Error: no supported images found")
        sys.exit(1)

    if len(image_files) < MIN_IMAGES:
        print(f"Warning: only {len(image_files)} images (recommend {MIN_IMAGES}+)")

    # Validate resolution
    from PIL import Image
    for img_path in image_files:
        with Image.open(img_path) as img:
            w, h = img.size
            if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
                print(f"Warning: {img_path.name} is {w}x{h} (min {MIN_RESOLUTION})")

    # Load config for caption prefix
    project_dir = Path(__file__).parent
    config_path = project_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    trigger = config.get("trigger_word", "cybrn")
    template = config.get("caption_template", "a painting in the style of {trigger}, ")
    caption_prefix = template.replace("{trigger}", trigger)

    print(f"Found {len(image_files)} images")
    print(f"Caption prefix: {caption_prefix}")
    print(f"Uploading to Modal Volume + computing CLIP embeddings...")

    image_bytes_list = [f.read_bytes() for f in image_files]
    filenames = [f.name for f in image_files]

    n = compute_embeddings.remote(image_bytes_list, filenames, caption_prefix)

    print(f"\nDone! {n} reference embeddings computed.")
    print(f"Training data uploaded to Modal Volume.")
    if dataset_name:
        print(f"Using HF dataset: {dataset_name}")
        print(f"Run: modal run train.py --dataset-name {dataset_name}")
    else:
        print(f"Run: modal run train.py")
