"""CLIP scoring module for autonomous SDXL LoRA training.

Provides cosine similarity, centroid scoring, nearest-neighbor scoring,
score aggregation, and CLIP image embedding via transformers.

Ported from autoresearch-lora (mlx_clip → PyTorch transformers CLIP).
Runs on Modal GPU alongside training.
"""

from pathlib import Path

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of embedding vectors."""
    return np.mean(embeddings, axis=0)


def score_against_centroid(eval_embedding: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine similarity of one eval image against the reference centroid."""
    return cosine_similarity(eval_embedding, centroid)


def score_nearest_neighbor(
    eval_embedding: np.ndarray, ref_embeddings: np.ndarray
) -> float:
    """Max cosine similarity of eval image against any reference image."""
    sims = [cosine_similarity(eval_embedding, ref) for ref in ref_embeddings]
    return max(sims)


def aggregate_scores(
    centroid_sims: list[float],
    nn_sims: list[float],
    neg_sims: list[float],
    num_prompts: int,
    seeds_per_prompt: int,
) -> dict:
    """Aggregate per-image scores into experiment-level metrics."""
    prompt_scores = []
    for i in range(num_prompts):
        start = i * seeds_per_prompt
        end = start + seeds_per_prompt
        prompt_scores.append(float(np.mean(centroid_sims[start:end])))

    return {
        "clip_sim_centroid": float(np.mean(centroid_sims)),
        "clip_sim_nn": float(np.mean(nn_sims)),
        "prompt_scores": prompt_scores,
        "score_stddev": float(np.std(centroid_sims)),
        "neg_control": float(np.mean(neg_sims)),
    }


# --- CLIP embedding (PyTorch transformers) ---

_clip_model = None
_clip_processor = None


def load_clip():
    """Load CLIP model (cached singleton)."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-large-patch14"
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _clip_model.eval()
    return _clip_model, _clip_processor


def embed_image(image_path: Path) -> np.ndarray:
    """Compute CLIP embedding for a single image. Returns numpy array."""
    import torch
    from PIL import Image

    model, processor = load_clip()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().flatten()
