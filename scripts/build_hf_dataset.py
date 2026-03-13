"""
Caption images with BLIP-2 on Modal and push to HuggingFace Hub.

This runs captioning on Modal (GPU) so you don't need a local GPU.
After captioning, it creates a HuggingFace dataset and pushes it,
so you can use --dataset-name directly in modal_train.py.

Usage:
    # Caption + push to HF Hub
    python scripts/build_hf_dataset.py \
        --input data/cyber-renaissance \
        --repo-id your-username/cyber-renaissance

    # Caption only (save captions locally, don't push)
    python scripts/build_hf_dataset.py \
        --input data/cyber-renaissance \
        --captions-only

    # With a style prefix added to every caption
    python scripts/build_hf_dataset.py \
        --input data/cyber-renaissance \
        --repo-id your-username/cyber-renaissance \
        --prefix "in the style of cyber renaissance, "

    # Edit captions before pushing (saves .txt files for manual review)
    python scripts/build_hf_dataset.py \
        --input data/cyber-renaissance \
        --repo-id your-username/cyber-renaissance \
        --review
"""
import argparse
import json
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal setup for BLIP-2 captioning
# ---------------------------------------------------------------------------

caption_image_modal = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch==2.5.1",
    "torchvision==0.20.1",
    "transformers==4.44.0",
    "accelerate==0.33.0",
    "Pillow>=10.0.0",
    "datasets==3.0.0",
    "huggingface-hub==0.36.0",
    "bitsandbytes==0.44.1",
)

app = modal.App("sdxl-dataset-builder", image=caption_image_modal)


@app.function(
    gpu="A10G",
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def caption_batch(image_bytes_list: list[bytes]) -> list[str]:
    """Caption a batch of images using BLIP-2 on GPU."""
    import io
    import torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    model_id = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    captions = []
    for img_bytes in image_bytes_list:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=75)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions.append(caption)

    return captions


@app.function(
    timeout=10 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_to_hub(image_bytes_list: list[bytes], captions: list[str], filenames: list[str], repo_id: str):
    """Create a HuggingFace dataset and push it."""
    import io
    from datasets import Dataset, Features, Value, Image as HFImage
    from PIL import Image

    images_pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]

    dataset = Dataset.from_dict(
        {"image": images_pil, "text": captions},
        features=Features({"image": HFImage(), "text": Value("string")}),
    )

    dataset.push_to_hub(repo_id, private=True)
    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")
    return repo_id


@app.local_entrypoint()
def main(
    input: str,
    repo_id: str = "",
    prefix: str = "",
    captions_only: bool = False,
    review: bool = False,
    batch_size: int = 8,
):
    input_dir = Path(input)
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in image_exts])

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")

    # Check for existing caption .txt files
    existing_captions = {}
    for img_file in image_files:
        txt_file = img_file.with_suffix(".txt")
        if txt_file.exists():
            existing_captions[img_file.name] = txt_file.read_text().strip()

    if existing_captions:
        print(f"Found {len(existing_captions)} existing caption files — using those")

    # Caption images that don't have .txt files yet
    needs_captioning = [f for f in image_files if f.name not in existing_captions]

    if needs_captioning:
        print(f"Captioning {len(needs_captioning)} images with BLIP-2 on Modal...")
        all_bytes = [f.read_bytes() for f in needs_captioning]

        # Process in batches
        new_captions = []
        for i in range(0, len(all_bytes), batch_size):
            batch = all_bytes[i : i + batch_size]
            print(f"  Batch {i // batch_size + 1}/{(len(all_bytes) + batch_size - 1) // batch_size}...")
            result = caption_batch.remote(batch)
            new_captions.extend(result)

        # Save caption .txt files alongside images
        for img_file, caption in zip(needs_captioning, new_captions):
            if prefix:
                caption = prefix + caption
            txt_file = img_file.with_suffix(".txt")
            txt_file.write_text(caption)
            existing_captions[img_file.name] = caption
            print(f"  {img_file.name}: {caption}")

    # Apply prefix to existing captions if specified
    if prefix:
        for img_file in image_files:
            cap = existing_captions.get(img_file.name, "")
            if cap and not cap.startswith(prefix):
                cap = prefix + cap
                existing_captions[img_file.name] = cap
                img_file.with_suffix(".txt").write_text(cap)

    # Build final caption list in image order
    captions = [existing_captions[f.name] for f in image_files]

    print(f"\n--- Captioning complete ({len(captions)} images) ---")

    if review:
        print("\nCaption .txt files saved alongside images. Edit them manually, then re-run without --review.")
        print("Caption files:")
        for f in image_files:
            print(f"  {f.with_suffix('.txt')}")
        return

    if captions_only:
        print("Captions saved as .txt files. Done.")
        return

    if not repo_id:
        print("No --repo-id specified. Captions saved locally. To push, re-run with --repo-id YOUR_USER/DATASET_NAME")
        return

    # Push to HuggingFace Hub
    print(f"\nPushing dataset to {repo_id}...")
    all_bytes = [f.read_bytes() for f in image_files]
    filenames = [f.name for f in image_files]
    push_to_hub.remote(all_bytes, captions, filenames, repo_id)

    print(f"\nDone! Use in training:")
    print(f"  modal run modal_train.py --exp-id 003 --dataset-name {repo_id}")
