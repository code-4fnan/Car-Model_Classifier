"""
Download car images from Wikimedia Commons, validate them, and split them
into train/ and val/ folders.

Usage:
    python prepare_data.py
    python prepare_data.py --per-class 120 --val-ratio 0.2 --seed 42

The script writes a CSV manifest containing source pages and license metadata.
Please inspect images and labels before training; Commons search results are
not guaranteed to be perfectly labelled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import re
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "my_dataset"
RAW_DIR = DATASET_DIR / "_download_cache"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "CarClassifierEducationalProject/1.0 (local student project)"

# Search expressions are deliberately specific; still review results manually.
CLASSES = {
    "supercars": [
        "supercar",
        "Ferrari supercar",
        "Lamborghini supercar",
        "McLaren supercar",
        "Bugatti supercar",
        "Porsche sports car",
    ],
    "suv_cars": [
        "SUV sport utility vehicle",
        "SUV car",
        "Range Rover SUV",
        "Jeep Wrangler SUV",
        "Toyota Land Cruiser SUV",
        "Ford SUV",
    ],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return value[:70] or "image"


def request_with_retries(
    session: requests.Session, url: str, *, params: dict | None = None,
    stream: bool = False, attempts: int = 4, timeout: int = 30
) -> requests.Response:
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(
                url, params=params, timeout=timeout, stream=stream
            )
            if response.status_code in (429, 500, 502, 503, 504):
                response.close()
                raise requests.HTTPError(
                    f"Temporary HTTP status {response.status_code}"
                )
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.Timeout) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {attempts} attempts: {last_error}")


def search_commons(
    session: requests.Session, phrase: str, limit: int
) -> list[dict[str, Any]]:
    """Find file pages and retrieve direct URLs plus attribution metadata."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": phrase,
        "gsrnamespace": "6",
        "gsrlimit": min(max(limit, 1), 50),
        "gsrwhat": "text",
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": "900",
    }
    response = request_with_retries(session, API_URL, params=params)
    payload = response.json()
    pages = payload.get("query", {}).get("pages", [])
    found = []
    for page in pages:
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        mime = (info.get("mime") or "").lower()
        if not mime.startswith("image/"):
            continue
        direct_url = info.get("thumburl") or info.get("url")
        if not direct_url:
            continue
        extmeta = info.get("extmetadata") or {}

        def meta(key: str) -> str:
            value = extmeta.get(key, {})
            if isinstance(value, dict):
                return re.sub(r"<[^>]+>", "", str(value.get("value", ""))).strip()
            return str(value)

        found.append({
            "title": page.get("title", ""),
            "page_url": "https://commons.wikimedia.org/wiki/" +
                        str(page.get("title", "")).replace(" ", "_"),
            "direct_url": direct_url,
            "mime": mime,
            "description": meta("ImageDescription"),
            "artist": meta("Artist"),
            "license": meta("LicenseShortName") or meta("UsageTerms"),
            "credit": meta("Credit"),
        })
    return found


def download_image(
    session: requests.Session, item: dict[str, Any], destination: Path
) -> tuple[bool, str]:
    """Download, fully decode, convert to RGB JPEG; reject invalid files."""
    try:
        response = request_with_retries(
            session, item["direct_url"], stream=False, timeout=45
        )
        content_type = response.headers.get("Content-Type", "").lower()
        data = response.content
        if not data or len(data) > 25 * 1024 * 1024:
            return False, "empty file or over 25 MB"
        if content_type and not (
            content_type.startswith("image/") or
            "octet-stream" in content_type
        ):
            return False, f"unexpected content type: {content_type}"

        # Verify first, then reopen to decode and normalize.
        with Image.open(io.BytesIO(data)) as check:
            check.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rgb = im.convert("RGB")
            if rgb.width < 100 or rgb.height < 100:
                return False, "image too small (minimum 100x100)"
            destination.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(destination, format="JPEG", quality=92, optimize=True)
        return True, hashlib.sha256(data).hexdigest()
    except (RuntimeError, requests.RequestException, OSError,
            UnidentifiedImageError, ValueError) as exc:
        return False, str(exc)


def split_files(
    files_by_class: dict[str, list[Path]], val_ratio: float, seed: int
) -> dict[str, dict[str, int]]:
    """Shuffle images deterministically and move 20% into val by default."""
    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = {}
    for class_name, paths in files_by_class.items():
        rng.shuffle(paths)
        if len(paths) < 2:
            raise RuntimeError(
                f"Only {len(paths)} valid image(s) for {class_name}. "
                "Run again or reduce --per-class after checking network access."
            )
        val_count = max(1, round(len(paths) * val_ratio))
        val_count = min(val_count, len(paths) - 1)
        validation = paths[:val_count]
        training = paths[val_count:]
        for split, split_paths in (("train", training), ("val", validation)):
            target_dir = DATASET_DIR / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source in split_paths:
                target = target_dir / source.name
                if source.resolve() != target.resolve():
                    source.replace(target)
        counts[class_name] = {"train": len(training), "val": len(validation)}
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=80,
                        help="Target valid downloads per class (default: 80)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation fraction between 0.1 and 0.4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.per_class < 4:
        parser.error("--per-class must be at least 4")
    if not 0.1 <= args.val_ratio <= 0.4:
        parser.error("--val-ratio must be between 0.1 and 0.4")

    # Remove previous generated splits to avoid stale/duplicate images.
    for split in ("train", "val"):
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            import shutil
            shutil.rmtree(split_dir)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    seen_hashes: set[str] = set()
    manifest_rows: list[dict[str, str]] = []
    files_by_class: dict[str, list[Path]] = {}

    for class_name, phrases in CLASSES.items():
        print(f"\n=== Collecting class: {class_name} ===")
        class_cache = RAW_DIR / class_name
        class_cache.mkdir(parents=True, exist_ok=True)
        accepted: list[Path] = []
        seen_titles: set[str] = set()
        candidate_limit = min(50, max(20, args.per_class))
        candidates: list[dict[str, Any]] = []

        for phrase in phrases:
            if len(candidates) >= args.per_class * 4:
                break
            try:
                results = search_commons(session, phrase, candidate_limit)
                print(f"Search '{phrase}': {len(results)} image candidates")
                for item in results:
                    title = item["title"]
                    if title not in seen_titles:
                        seen_titles.add(title)
                        item["search_phrase"] = phrase
                        candidates.append(item)
            except (RuntimeError, ValueError, requests.RequestException) as exc:
                print(f"Search warning for '{phrase}': {exc}")
            time.sleep(0.4)

        for index, item in enumerate(candidates, start=1):
            if len(accepted) >= args.per_class:
                break
            filename = f"{class_name}_{index:04d}_{safe_name(item['title'])}.jpg"
            destination = class_cache / filename
            ok, digest_or_reason = download_image(session, item, destination)
            if not ok:
                print(f"  Skipped {item['title']}: {digest_or_reason}")
                continue
            digest = digest_or_reason
            if digest in seen_hashes:
                destination.unlink(missing_ok=True)
                continue
            seen_hashes.add(digest)
            accepted.append(destination)
            manifest_rows.append({
                "class": class_name,
                "filename": filename,
                "title": item["title"],
                "source_page": item["page_url"],
                "license": item["license"],
                "artist": item["artist"],
                "credit": item["credit"],
                "description": item["description"],
                "search_phrase": item["search_phrase"],
                "sha256": digest,
            })
            print(f"  Downloaded {len(accepted)}/{args.per_class}: {item['title']}")
            time.sleep(0.15)

        files_by_class[class_name] = accepted
        print(f"Valid unique images for {class_name}: {len(accepted)}")

    if any(len(paths) < 2 for paths in files_by_class.values()):
        raise SystemExit(
            "\nNot enough valid images were downloaded for both classes. "
            "Check your internet connection, then rerun prepare_data.py. "
            "You can also lower --per-class, e.g. --per-class 20."
        )

    counts = split_files(files_by_class, args.val_ratio, args.seed)

    # Keep an attribution manifest, including each file's Commons source page.
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        columns = [
            "class", "filename", "source_page", "license", "artist", "credit",
            "description", "search_phrase", "sha256"
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nDataset preparation complete.")
    print(f"Dataset root: {DATASET_DIR}")
    print(f"Attribution manifest: {MANIFEST_PATH}")
    for class_name, split_counts in counts.items():
        print(f"  {class_name}: train={split_counts['train']}, "
              f"val={split_counts['val']}")
    print("\nPlease inspect my_dataset/ and manifest.csv before training.")
    print("Image results can still be mislabeled despite search filtering.")


if __name__ == "__main__":
    main()
