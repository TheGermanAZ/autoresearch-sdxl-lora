"""
Download and organize images for LoRA training.

Supports three input modes:
  1. A text file with one image URL per line
  2. gallery-dl to download from ArtStation, DeviantArt, etc.
  3. A local folder of images to process

Usage:
    # From a URL list
    python scripts/collect_images.py --urls urls.txt --output data/cyber-renaissance

    # From gallery-dl (downloads a user's gallery or search tag)
    python scripts/collect_images.py --gallery-dl "https://www.artstation.com/search?query=cyber+renaissance" --output data/cyber-renaissance

    # Process existing local folder (resize, validate, deduplicate)
    python scripts/collect_images.py --input-dir ~/Downloads/my-images --output data/cyber-renaissance

    # With custom resolution
    python scripts/collect_images.py --urls urls.txt --output data/cyber-renaissance --resolution 1024
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


def download_from_urls(url_file: Path, output_dir: Path) -> list[Path]:
    """Download images from a text file of URLs (one per line)."""
    urls = [line.strip() for line in url_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
    print(f"Found {len(urls)} URLs in {url_file}")

    downloaded = []
    for i, url in enumerate(urls):
        ext = Path(url.split("?")[0]).suffix or ".jpg"
        dest = output_dir / f"img_{i:04d}{ext}"
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            downloaded.append(dest)
            print(f"  [{i+1}/{len(urls)}] {dest.name}")
        except Exception as e:
            print(f"  [{i+1}/{len(urls)}] FAILED: {e}")

    return downloaded


def download_with_gallery_dl(source: str, output_dir: Path) -> list[Path]:
    """Use gallery-dl to download images from a URL."""
    if shutil.which("gallery-dl") is None:
        print("gallery-dl not found. Install with: pip install gallery-dl")
        sys.exit(1)

    cmd = ["gallery-dl", "--dest", str(output_dir), source]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    return list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.png")) + list(output_dir.rglob("*.webp"))


def copy_from_local(input_dir: Path, output_dir: Path) -> list[Path]:
    """Copy images from a local directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    sources = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]
    print(f"Found {len(sources)} images in {input_dir}")

    copied = []
    for i, src in enumerate(sorted(sources)):
        dest = output_dir / f"img_{i:04d}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        copied.append(dest)

    return copied


def validate_and_resize(images: list[Path], resolution: int, min_size: int = 512) -> list[Path]:
    """Validate images, resize to target resolution, discard broken/tiny ones."""
    valid = []
    for img_path in images:
        try:
            img = Image.open(img_path)
            img.verify()
            img = Image.open(img_path)  # reopen after verify

            w, h = img.size
            if min(w, h) < min_size:
                print(f"  Skipping {img_path.name}: too small ({w}x{h})")
                img_path.unlink()
                continue

            # Resize so the short side = resolution, then center crop
            scale = resolution / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            left = (new_w - resolution) // 2
            top = (new_h - resolution) // 2
            img = img.crop((left, top, left + resolution, top + resolution))

            # Save as PNG for lossless quality
            out_path = img_path.with_suffix(".png")
            img.convert("RGB").save(out_path, "PNG")
            if out_path != img_path:
                img_path.unlink()

            valid.append(out_path)
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
            img_path.unlink()

    return valid


def deduplicate(images: list[Path]) -> list[Path]:
    """Remove exact duplicate images by file hash."""
    seen = {}
    unique = []
    for img_path in images:
        h = hashlib.md5(img_path.read_bytes()).hexdigest()
        if h in seen:
            print(f"  Duplicate: {img_path.name} == {seen[h].name}")
            img_path.unlink()
        else:
            seen[h] = img_path
            unique.append(img_path)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Collect and prepare images for LoRA training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--urls", type=Path, help="Text file with one image URL per line")
    group.add_argument("--gallery-dl", dest="gallery_dl", type=str, help="URL for gallery-dl to download")
    group.add_argument("--input-dir", dest="input_dir", type=Path, help="Local folder of images to process")

    parser.add_argument("--output", type=Path, required=True, help="Output directory for processed images")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution (default: 1024)")
    parser.add_argument("--min-size", dest="min_size", type=int, default=512, help="Min image dimension to keep (default: 512)")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1: Get raw images
    if args.urls:
        images = download_from_urls(args.urls, args.output)
    elif args.gallery_dl:
        images = download_with_gallery_dl(args.gallery_dl, args.output)
    else:
        images = copy_from_local(args.input_dir, args.output)

    if not images:
        print("No images found.")
        sys.exit(1)

    # Step 2: Validate and resize
    print(f"\nValidating and resizing {len(images)} images to {args.resolution}x{args.resolution}...")
    images = validate_and_resize(images, args.resolution, args.min_size)

    # Step 3: Deduplicate
    print(f"\nDeduplicating {len(images)} images...")
    images = deduplicate(images)

    print(f"\nDone! {len(images)} images ready in {args.output}")
    print(f"Next step: python scripts/build_hf_dataset.py --input {args.output} --repo-id YOUR_USERNAME/cyber-renaissance")


if __name__ == "__main__":
    main()
