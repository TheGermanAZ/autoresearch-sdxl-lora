"""
Generate synthetic cyber renaissance training images using base SDXL on Modal.

Creates a diverse dataset by generating images from varied prompts that
blend renaissance painting aesthetics with cyberpunk elements.

Usage:
    # Generate full dataset (100 images)
    modal run scripts/generate_dataset.py

    # Generate a smaller test batch
    modal run scripts/generate_dataset.py --count 10

    # Custom output dir
    modal run scripts/generate_dataset.py --output-dir data/cyber-renaissance-v2
"""
import os
import modal

MINUTES = 60

volume = modal.Volume.from_name("sdxl-lora-experiments", create_if_missing=True)
VOL_DIR = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
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

app = modal.App("sdxl-dataset-gen", image=image)

# ---------------------------------------------------------------------------
# Prompt templates — mix subjects, compositions, and fusion elements
# ---------------------------------------------------------------------------

SUBJECTS = [
    "a noblewoman with cybernetic implants visible beneath her skin",
    "a knight in chrome-plated armor with glowing blue joints",
    "an elderly scholar with holographic monocle and mechanical hand",
    "a young prince with fiber-optic hair and LED-lit crown",
    "a cardinal in red robes with circuit-board embroidery",
    "a merchant examining a holographic gem with a brass magnifying lens",
    "twin courtesans with mirrored cybernetic faces",
    "an armored angel with mechanical wings and a plasma halo",
    "a plague doctor with a neon gas mask and data-readout lenses",
    "a Renaissance madonna cradling a small android infant",
]

COMPOSITIONS = [
    "portrait in the style of Caravaggio, dramatic chiaroscuro lighting",
    "three-quarter view portrait in the style of Vermeer, soft window light",
    "full body posed like a Raphael fresco, architectural background",
    "bust portrait in the style of Da Vinci, sfumato technique",
    "group scene in the style of Rembrandt's Night Watch",
    "reclining pose in the style of Titian, rich velvet draping",
    "profile portrait in the style of Botticelli, flowing composition",
    "seated portrait in the style of Velázquez, regal bearing",
    "close-up face study in the style of Antonello da Messina",
    "standing figure in contrapposto, Michelangelo-inspired anatomy",
]

ENVIRONMENTS = [
    "in a marble palace with holographic stained glass windows",
    "in a candlelit cathedral with floating data streams above the altar",
    "in a Renaissance courtyard with neon vines climbing the columns",
    "in a dark study filled with glowing manuscripts and chrome instruments",
    "in a grand hall with cybernetic cherubs carved into the ceiling",
    "against a twilight sky with circuit-board constellations",
    "in a garden with bioluminescent flowers and mechanical butterflies",
    "in an alchemist's lab with holographic flask contents",
    "on a baroque balcony overlooking a neon-lit Renaissance city",
    "in a throne room where the walls pulse with embedded circuitry",
]

STYLE_SUFFIX = (
    "oil painting texture, visible brushstrokes, "
    "neon accents bleeding through classical palette, "
    "renaissance composition with cyberpunk elements, "
    "masterful blend of old masters technique and futuristic technology, "
    "highly detailed, 8k"
)

NEGATIVE_PROMPT = (
    "ugly, blurry, low quality, deformed, disfigured, "
    "cartoon, anime, 3d render, photograph, modern clothing, "
    "plain background, simple, flat colors, watermark, text"
)

# Additional standalone prompts for variety (still life, architecture, etc.)
STANDALONE_PROMPTS = [
    "a baroque still life of silver fruit, crystal goblets, and hovering holographic butterflies on a dark velvet tablecloth, oil painting with neon rim lighting",
    "a grand Renaissance cathedral interior where the stained glass windows display scrolling code and the candles are replaced by plasma lights, oil painting style",
    "a cyber-renaissance landscape of rolling Tuscan hills dotted with chrome spires and holographic billboards among cypress trees, golden hour, oil painting",
    "an ornate golden picture frame containing a portal to a neon cyberpunk city, hanging on a cracked marble wall, dramatic lighting, oil painting texture",
    "a Renaissance workshop table covered in brass gears, glowing vials, a mechanical hand sketch by Da Vinci, and a hovering holographic blueprint, chiaroscuro",
    "a baroque vanitas painting with a chrome skull, wilting flowers with LED veins, an hourglass filled with glowing data particles, dramatic side lighting",
    "a grand staircase in a Renaissance palace where each marble step pulses with embedded circuitry, oil painting with neon accents",
    "a cyber-renaissance ceiling fresco depicting gods and angels with mechanical wings and holographic halos, Sistine Chapel composition, ornate gold frame",
    "a jousting tournament in a neon-lit arena, knights in chrome plate armor on mechanical horses, Renaissance crowd in balconies, oil painting style",
    "a Renaissance map room with holographic globes, brass telescopes with digital readouts, and star charts projected on vaulted ceilings, candlelight and neon",
    "a baroque ship in a bottle but the ship is cybernetic with glowing thrusters, surrounded by miniature holographic waves, dark background, oil painting",
    "a Renaissance garden maze viewed from above, hedges interwoven with glowing fiber optic threads, marble fountains with holographic water, twilight",
    "a Medici-style banquet hall with a long table of chrome and crystal, holographic feast, robotic servants in period costume, candlelight and neon",
    "an old master's self-portrait where half the face is human with oil paint texture and half is chrome cybernetic with glowing eye, dramatic lighting",
    "a Renaissance apothecary shop with shelves of glowing potions in ornate bottles, a mechanical owl perched on a brass stand, warm candlelight with neon accents",
    "a baroque music room with a harpsichord made of chrome and wood, holographic sheet music floating in the air, rich velvet curtains, oil painting",
    "a Renaissance courtyard fountain where the water is replaced by cascading light particles, marble cherubs with circuit patterns, golden hour",
    "a classical painting of hands reaching toward each other, one human with oil paint texture and one chrome cybernetic, inspired by the Creation of Adam",
    "a cyber-renaissance armor display in a grand hall, chrome suits with glowing joints on marble pedestals, dramatic chiaroscuro lighting, oil painting",
    "a Renaissance library with towering bookshelves where some books glow with holographic pages, brass reading lamps with neon filaments, warm atmosphere",
]


def build_prompt_list(count: int) -> list[str]:
    """Build a list of diverse prompts by combining templates + standalone prompts."""
    prompts = []

    # Combinatorial prompts from subject x composition x environment
    import itertools
    combos = list(itertools.product(SUBJECTS, COMPOSITIONS, ENVIRONMENTS))
    import random
    random.seed(42)
    random.shuffle(combos)

    for subject, composition, environment in combos[:max(0, count - len(STANDALONE_PROMPTS))]:
        prompt = f"{subject}, {composition}, {environment}, {STYLE_SUFFIX}"
        prompts.append(prompt)

    # Add standalone prompts
    for p in STANDALONE_PROMPTS:
        prompts.append(f"{p}, {STYLE_SUFFIX}")

    random.shuffle(prompts)
    return prompts[:count]


@app.function(
    gpu="A100-80GB",
    volumes={VOL_DIR: volume},
    timeout=60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_batch(prompts: list[str], start_idx: int, output_dir: str, negative_prompt: str):
    """Generate a batch of images on GPU."""
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")

    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        idx = start_idx + i
        seed = 42 + idx  # deterministic but varied

        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]

        img_path = f"{output_dir}/img_{idx:04d}.png"
        image.save(img_path)

        # Save caption alongside
        caption_path = f"{output_dir}/img_{idx:04d}.txt"
        with open(caption_path, "w") as f:
            f.write(prompt)

        print(f"[{idx:04d}] saved — seed={seed}")

    # Note: no volume.commit() here — runs via .map() and concurrent commits
    # can overwrite each other's snapshots. Modal auto-persists writes.
    return len(prompts)


@app.local_entrypoint()
def main(count: int = 100, output_dir: str = "dataset/cyber-renaissance", batch_size: int = 10):
    prompts = build_prompt_list(count)
    remote_dir = f"{VOL_DIR}/{output_dir}"

    print(f"Generating {len(prompts)} cyber renaissance images...")
    print(f"Output: {remote_dir}")
    print(f"Batch size: {batch_size}")

    # Build parallel batch args
    batch_prompts = []
    batch_starts = []
    batch_dirs = []
    batch_negs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts.append(prompts[i : i + batch_size])
        batch_starts.append(i)
        batch_dirs.append(remote_dir)
        batch_negs.append(NEGATIVE_PROMPT)

    print(f"Launching {len(batch_prompts)} batches in parallel on separate GPUs...\n")

    # Fan out to parallel GPUs
    results = list(generate_batch.map(batch_prompts, batch_starts, batch_dirs, batch_negs))
    total = sum(results)

    print(f"\nDone! {total} images generated on Modal volume.")
    print(f"Download with: modal volume get sdxl-lora-experiments {output_dir} data/cyber-renaissance")
