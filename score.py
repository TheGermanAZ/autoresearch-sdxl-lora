"""Multi-metric scoring module for autonomous SDXL LoRA training.

Scorers:
  - CLIP centroid/NN: style similarity to reference images
  - PickScore: human preference alignment (image-text)
  - HPSv2: aesthetic quality based on human preference data
  - VLM Judge: vision LLM qualitative assessment (optional, API call)

All GPU scorers run on Modal alongside training.
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of embedding vectors."""
    return np.mean(embeddings, axis=0)


# ---------------------------------------------------------------------------
# CLIP scoring (style similarity)
# ---------------------------------------------------------------------------

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
    """Compute CLIP embedding for a single image."""
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


def score_against_centroid(eval_embedding: np.ndarray, centroid: np.ndarray) -> float:
    return cosine_similarity(eval_embedding, centroid)


def score_nearest_neighbor(eval_embedding: np.ndarray, ref_embeddings: np.ndarray) -> float:
    sims = [cosine_similarity(eval_embedding, ref) for ref in ref_embeddings]
    return max(sims)


# ---------------------------------------------------------------------------
# PickScore (human preference for image-text alignment)
# ---------------------------------------------------------------------------

_pick_model = None
_pick_processor = None


def load_pickscore():
    """Load PickScore model (cached singleton)."""
    global _pick_model, _pick_processor
    if _pick_model is None:
        import torch
        from transformers import AutoModel, AutoProcessor

        model_id = "yuvalkirstain/PickScore_v1"
        _pick_processor = AutoProcessor.from_pretrained(model_id)
        _pick_model = AutoModel.from_pretrained(model_id).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _pick_model.eval()
    return _pick_model, _pick_processor


def pickscore(image_path: Path, prompt: str) -> float:
    """Score image-text alignment using PickScore. Higher = better."""
    import torch
    from PIL import Image

    model, processor = load_pickscore()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        images=image, text=prompt, return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    ).to(device)

    with torch.no_grad():
        image_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        score = (image_emb * text_emb).sum(dim=-1).item()

    return score


# ---------------------------------------------------------------------------
# HPSv2 (aesthetic quality from human preferences)
# ---------------------------------------------------------------------------

_hps_model = None
_hps_processor = None


def load_hpsv2():
    """Load HPSv2 model (cached singleton). Uses CLIP fine-tuned on human preference data."""
    global _hps_model, _hps_processor
    if _hps_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model_id = "adams-story/HPSv2"
        _hps_processor = CLIPProcessor.from_pretrained(model_id)
        _hps_model = CLIPModel.from_pretrained(model_id).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        _hps_model.eval()
    return _hps_model, _hps_processor


def hpsv2_score(image_path: Path, prompt: str) -> float:
    """Score aesthetic quality using HPSv2. Higher = better."""
    import torch
    from PIL import Image

    model, processor = load_hpsv2()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        images=image, text=prompt, return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits_per_image.item() / 100.0  # normalize

    return score


# ---------------------------------------------------------------------------
# VLM Judge (vision LLM qualitative assessment)
# ---------------------------------------------------------------------------

VLM_JUDGE_RUBRIC = """Rate this image on a scale of 1-10 for each criterion:

1. **Style fusion quality**: How well does it blend renaissance painting aesthetics (composition, brushwork, chiaroscuro) with cyberpunk elements (neon, chrome, circuitry)?
2. **Technical quality**: Is the image free of artifacts, distortions, or incoherent elements?
3. **Aesthetic appeal**: Is the image visually striking and cohesive?

Respond with ONLY three numbers separated by commas, nothing else. Example: 7,8,6"""


def vlm_judge(image_path: Path, api_key: str = None) -> dict:
    """Score image using Gemini 1.5 Pro via OpenRouter. Returns per-criterion scores.

    Requires OPENROUTER_API_KEY environment variable or api_key parameter.
    Cost: ~$0.002-0.005 per image (much cheaper than Claude vision).
    """
    import base64
    import os
    import json
    import re
    from urllib.request import Request, urlopen

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return {"style_fusion": 0.0, "technical": 0.0, "aesthetic": 0.0, "vlm_avg": 0.0}

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    suffix = str(image_path).lower()
    media_type = "image/png" if suffix.endswith(".png") else "image/jpeg"

    payload = json.dumps({
        "model": "google/gemini-3-pro",
        "max_tokens": 50,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_b64}"}},
                {"type": "text", "text": VLM_JUDGE_RUBRIC},
            ],
        }],
    })

    req = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload.encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )

    try:
        with urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        text = result["choices"][0]["message"]["content"].strip()
        nums = re.findall(r"(\d+)", text)
        if len(nums) >= 3:
            style = float(nums[0]) / 10.0
            technical = float(nums[1]) / 10.0
            aesthetic = float(nums[2]) / 10.0
            return {
                "style_fusion": style,
                "technical": technical,
                "aesthetic": aesthetic,
                "vlm_avg": (style + technical + aesthetic) / 3.0,
            }
    except Exception as e:
        print(f"  VLM judge failed for {image_path.name}: {e}")

    return {"style_fusion": 0.0, "technical": 0.0, "aesthetic": 0.0, "vlm_avg": 0.0}


# ---------------------------------------------------------------------------
# Multi-metric aggregation
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "clip_centroid": 0.25,
    "pickscore": 0.25,
    "hpsv2": 0.25,
    "vlm_avg": 0.25,
}

WEIGHTS_NO_VLM = {
    "clip_centroid": 0.35,
    "pickscore": 0.35,
    "hpsv2": 0.30,
}


def aggregate_multi_scores(
    clip_centroid_sims: list[float],
    clip_nn_sims: list[float],
    neg_sims: list[float],
    pickscore_vals: list[float],
    hpsv2_vals: list[float],
    vlm_vals: list[float],
    num_prompts: int,
    seeds_per_prompt: int,
) -> dict:
    """Aggregate all metrics into a single run-level score."""
    # Per-prompt CLIP scores
    prompt_scores = []
    for i in range(num_prompts):
        start = i * seeds_per_prompt
        end = start + seeds_per_prompt
        prompt_scores.append(float(np.mean(clip_centroid_sims[start:end])))

    clip_centroid = float(np.mean(clip_centroid_sims))
    clip_nn = float(np.mean(clip_nn_sims))
    neg_control = float(np.mean(neg_sims)) if neg_sims else 0.0
    pick_avg = float(np.mean(pickscore_vals)) if pickscore_vals else 0.0
    hps_avg = float(np.mean(hpsv2_vals)) if hpsv2_vals else 0.0
    vlm_avg = float(np.mean(vlm_vals)) if vlm_vals else 0.0

    # Composite score
    has_vlm = vlm_avg > 0
    weights = DEFAULT_WEIGHTS if has_vlm else WEIGHTS_NO_VLM

    # Normalize scores to roughly [0, 1] for fair weighting
    # CLIP centroid is already ~[0, 1]
    # PickScore is typically ~[0.15, 0.30], scale to [0, 1]
    pick_norm = min(max((pick_avg - 0.15) / 0.15, 0), 1)
    # HPSv2 is typically ~[0.20, 0.35], scale to [0, 1]
    hps_norm = min(max((hps_avg - 0.20) / 0.15, 0), 1)

    if has_vlm:
        composite = (
            weights["clip_centroid"] * clip_centroid
            + weights["pickscore"] * pick_norm
            + weights["hpsv2"] * hps_norm
            + weights["vlm_avg"] * vlm_avg
        )
    else:
        composite = (
            weights["clip_centroid"] * clip_centroid
            + weights["pickscore"] * pick_norm
            + weights["hpsv2"] * hps_norm
        )

    return {
        "clip_sim_centroid": clip_centroid,
        "clip_sim_nn": clip_nn,
        "prompt_scores": prompt_scores,
        "score_stddev": float(np.std(clip_centroid_sims)),
        "neg_control": neg_control,
        "pickscore_avg": pick_avg,
        "hpsv2_avg": hps_avg,
        "vlm_avg": vlm_avg,
        "composite_score": float(composite),
    }
