#!/usr/bin/env python3
"""Download a small pool of outdoor Poly Haven HDRIs into resources/hdri.

Indoor worlds use these as the sky seen through windows. Outdoor / puresky maps
only — indoor studio HDRIs would look like the apartment sits inside another room.

Poly Haven is CC0. Cache locally; do not hit the API on every scene.

Usage (from infinigen/):
    python tools/download_polyhaven_hdris.py
    python tools/download_polyhaven_hdris.py --resolution 1k
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com"
USER_AGENT = "syndefect3d-hdri-download/1.0 (https://github.com/shiven2001/syndefect3d)"

# Outdoor / puresky maps that read as "outside a window", not studio lighting.
DEFAULT_IDS = [
    "kloppenheim_06_puresky",
    "kloofendal_43d_clear_puresky",
    "overcast_soil_puresky",
    "sunflowers_puresky",
    "evening_road_01_puresky",
    "syferfontein_18d_clear_puresky",
    "wasteland_clouds_puresky",
    "qwantani_puresky",
]


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def hdri_dir() -> Path:
    # tools/ is inside infinigen/; resources/hdri sits next to that package root.
    return Path(__file__).resolve().parents[1] / "resources" / "hdri"


def file_url(asset_id: str, resolution: str) -> tuple[str, str] | None:
    data = _get_json(f"{API}/files/{asset_id}")
    hdri = data.get("hdri") or {}
    res = hdri.get(resolution) or {}
    for fmt in ("exr", "hdr"):
        info = res.get(fmt)
        if info and info.get("url"):
            return info["url"], fmt
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        default="2k",
        choices=("1k", "2k", "4k"),
        help="HDRI resolution. 2k is enough for window fill.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=DEFAULT_IDS,
        help="Poly Haven asset ids (default: outdoor puresky set).",
    )
    args = parser.parse_args()

    out_dir = hdri_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for asset_id in args.ids:
        dest_exr = out_dir / f"{asset_id}_{args.resolution}.exr"
        dest_hdr = out_dir / f"{asset_id}_{args.resolution}.hdr"
        if dest_exr.exists() or dest_hdr.exists():
            existing = dest_exr if dest_exr.exists() else dest_hdr
            print(f"skip {asset_id} (already have {existing.name})")
            saved.append(existing.name)
            continue
        print(f"fetch {asset_id} {args.resolution}...")
        try:
            loc = file_url(asset_id, args.resolution)
        except Exception as exc:
            print(f"  fail listing {asset_id}: {exc}", file=sys.stderr)
            continue
        if loc is None:
            print(f"  no {args.resolution} exr/hdr for {asset_id}", file=sys.stderr)
            continue
        url, fmt = loc
        dest = out_dir / f"{asset_id}_{args.resolution}.{fmt}"
        try:
            _download(url, dest)
        except Exception as exc:
            print(f"  fail download {asset_id}: {exc}", file=sys.stderr)
            continue
        print(f"  wrote {dest}")
        saved.append(dest.name)

    manifest = {
        "source": "https://polyhaven.com/hdris",
        "license": "CC0",
        "resolution": args.resolution,
        "files": saved,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done: {len(saved)} files in {out_dir}")
    if not saved:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
