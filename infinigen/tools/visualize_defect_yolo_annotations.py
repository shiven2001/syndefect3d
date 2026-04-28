#!/usr/bin/env python3
"""
Sanity-check defect YOLO-style boxes **before** running prepare_defect_annotated_dataset.py.

Uses the same pipeline as that preparer on raw renders:
  - discover_scene_frames (RGB + MaterialSegmentation .npy + Materials .json)
  - compute_asset_bboxes_from_material_passes
  - YOLO text lines use the same center/size normalization as write_bboxes in the preparer

Example:
  python tools/visualize_defect_yolo_annotations.py -i /path/to/all_frames -o preview.png
  python tools/visualize_defect_yolo_annotations.py -i /path/to/all_frames --seed 0 --print-yolo
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from imageio import imread
from PIL import Image, ImageDraw, ImageFont

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import prepare_defect_annotated_dataset as prep  # noqa: E402

YOLO_CLASS_COLORS = [
    (220, 60, 60),
    (60, 160, 255),
    (80, 200, 80),
    (230, 180, 40),
    (180, 80, 220),
]


def yolo_line_from_box(
    b: dict, image_h: int, image_w: int
) -> str:
    """Same string as prepare_defect_annotated_dataset.write_bboxes would emit per box."""
    xc = (b["x_min"] + b["x_max"] + 1) / 2.0 / image_w
    yc = (b["y_min"] + b["y_max"] + 1) / 2.0 / image_h
    bw = b["width"] / image_w
    bh = b["height"] / image_h
    yolo_cls = b["class_id"] - 1
    return f"{yolo_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def load_frame_triplet(image_path: Path, npy_path: Path, json_path: Path):
    """Match prepare_defect_annotated_dataset.process_sample loading (RGB + index_map + materials)."""
    rgb = imread(image_path)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    index_map = np.load(npy_path)
    if index_map.ndim == 3:
        index_map = index_map.squeeze()
    from PIL import Image as PILImage

    if index_map.shape[:2] != rgb.shape[:2]:
        pil_mask = PILImage.fromarray(index_map.astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), PILImage.NEAREST
        )
        index_map = np.array(pil_mask)

    materials = json.loads(json_path.read_text())
    return rgb, index_map, materials


def draw_boxes(rgb: Image.Image, boxes: list[dict]) -> Image.Image:
    """Draw pixel boxes from compute_asset_bboxes_from_material_passes."""
    img = rgb.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for b in boxes:
        cid = int(b["class_id"])
        internal_idx = max(0, min(4, cid - 1))
        color = YOLO_CLASS_COLORS[internal_idx % len(YOLO_CLASS_COLORS)]
        x0, y0 = int(b["x_min"]), int(b["y_min"])
        x1, y1 = int(b["x_max"]), int(b["y_max"])
        x0 = max(0, min(w - 1, x0))
        x1 = max(0, min(w - 1, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(0, min(h - 1, y1))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, min(w, h) // 400))
        yc = cid - 1
        name = b.get("class_name", prep.CLASS_NAMES.get(cid, "?"))
        extra = f" pass={b.get('pass_index', '?')}" if "pass_index" in b else ""
        label = f"{yc}:{name}{extra}"
        if font:
            draw.text((x0 + 2, max(0, y0 - 10)), label, fill=color, font=font)
        else:
            draw.text((x0 + 2, max(0, y0 - 10)), label, fill=color)
    return img


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize defect boxes from raw Infinigen frames using the same logic as "
            "prepare_defect_annotated_dataset.py (material pass_index bboxes + YOLO lines)."
        )
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        type=Path,
        required=True,
        help=(
            "Root passed to discover_scene_frames (e.g. all_frames or a single scene/rig folder)."
        ),
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Specific sample id from discovery (e.g. livingroom01_rig0_camera_0). Default: random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for random sample selection.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("defect_yolo_preview.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--print-yolo",
        action="store_true",
        help="Print YOLO lines (same as bboxes_yolo/<id>.txt from the preparer) to stdout.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print bbox dict list as JSON (same content as bboxes/<id>.json 'boxes' key).",
    )
    args = parser.parse_args()

    root = args.input_folder.resolve()
    samples = prep.discover_scene_frames(root)
    if not samples:
        raise SystemExit(f"No frame triplets under {root} (need Image/, MaterialSegmentation/, Materials/).")

    rng = random.Random(args.seed)
    if args.sample_id is not None:
        chosen = None
        for img_path, npy_path, json_path, sid in samples:
            if sid == args.sample_id:
                chosen = (img_path, npy_path, json_path, sid)
                break
        if chosen is None:
            raise SystemExit(f"--sample-id {args.sample_id!r} not among {len(samples)} discovered samples.")
    else:
        chosen = rng.choice(samples)

    img_path, npy_path, json_path, sid = chosen
    rgb_np, index_map, materials = load_frame_triplet(img_path, npy_path, json_path)
    h, w = index_map.shape[:2]
    boxes = prep.compute_asset_bboxes_from_material_passes(
        index_map, materials, min_area=prep.BBOX_MIN_PIXEL_AREA
    )

    if args.print_json:
        print(json.dumps(boxes, indent=2))

    if args.print_yolo:
        print(f"# sample_id={sid}  (YOLO class 0..4 = defects; same as preparer bboxes_yolo/*.txt)")
        for b in boxes:
            print(yolo_line_from_box(b, h, w))

    pil = Image.fromarray(rgb_np.astype(np.uint8))
    out = draw_boxes(pil, boxes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"Wrote {args.output}  (sample_id={sid}, n_boxes={len(boxes)})")
    print(f"  RGB: {img_path}")
    print(f"  Seg: {npy_path}")
    print(f"  Mat: {json_path}")


if __name__ == "__main__":
    main()
