#!/usr/bin/env python3
"""
Prepare U-Net training data for defect segmentation from Infinigen render outputs.

Reads:
  - RGB images: .../Image/camera_*/Image_*_0_0001_0.png
  - Material index maps: .../MaterialSegmentation/camera_*/MaterialSegmentation_*_0_0001_0.npy
  - Material name -> pass_index: .../Materials/camera_*/Materials_*_0_0001_0.json

Layouts supported:
  - <input>/<scene>/frames/Image/... (standard Infinigen)
  - <input>/<scene>/Image/... (no frames/)
  - <input>/<scene>/<rig_dir>/Image/... e.g. all_frames/bathroom01/rig0_rs0/Image/...

Maps defect materials to classes:
  - 0: background (non-defect)
  - 1: CrackMaterial_*
  - 2: PaintPeelMaterial_*
  - 3: SpallingMaterial_*
  - 4: BubbleMaterial_*
  - 5: OpenWiringMaterial_*

Outputs:
  - out_dir/images/<id>.png  (RGB)
  - out_dir/masks/<id>.png   (single-channel uint8: class k -> k * MASK_GRAY_STEP, max <= 180;
                              visible in viewers; distinct from legacy 0,85,170,255)
  - out_dir/bboxes/<id>.json (one 2D bbox per defect *material pass* / placed asset plane:
                              union of all pixels with that Blender pass_index; pixel coords
                              {x_min,y_min,x_max,y_max}; fields pass_index, material_name)
  - out_dir/bboxes_yolo/<id>.txt (YOLO-format: class_id xc_n yc_n w_n h_n, all in [0,1])
  - out_dir/splits/train.txt, val.txt, test.txt (~70% / 15% / 15%, reproducible with --seed)
  - Existing image+mask pairs are skipped on re-run; splits are always refreshed from disk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from imageio import imread, imwrite
from tqdm import tqdm


# Class IDs for U-Net
CLASS_BACKGROUND = 0
CLASS_CRACK = 1
CLASS_PAINT_PEEL = 2
CLASS_SPALLING = 3
CLASS_PAINT_BUBBLE = 4
CLASS_EXPOSED_WIRING = 5

DEFECT_PREFIXES = {
    "CrackMaterial": CLASS_CRACK,
    "PaintPeelMaterial": CLASS_PAINT_PEEL,
    "SpallingMaterial": CLASS_SPALLING,
    "BubbleMaterial": CLASS_PAINT_BUBBLE,
    "OpenWiringMaterial": CLASS_EXPOSED_WIRING,
}

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_CRACK: "crack",
    CLASS_PAINT_PEEL: "paint_peel",
    CLASS_SPALLING: "spalling",
    CLASS_PAINT_BUBBLE: "paint_bubble",
    CLASS_EXPOSED_WIRING: "exposed_wiring",
}

NUM_CLASSES = 1 + max(DEFECT_PREFIXES.values())  # background 0 + defects 1..5 -> 6 classes
# PNG mask grayscale: class k -> k * step (step=36 for 6 classes). Keeps levels under legacy 85/170/255.
MASK_GRAY_STEP = max(1, 180 // (NUM_CLASSES - 1))

# Drop defect materials with almost no visible pixels in this frame (speckle / grazing).
BBOX_MIN_PIXEL_AREA = 4


def build_pass_index_to_class(materials_json: dict) -> dict[int, int]:
    """Build mapping from Blender pass_index to class id 0..NUM_CLASSES-1 (0=background)."""
    pass_to_class = {}
    for mat_name, info in materials_json.items():
        pass_index = info["pass_index"]
        cls = CLASS_BACKGROUND
        for prefix, class_id in DEFECT_PREFIXES.items():
            if mat_name.startswith(prefix):
                cls = class_id
                break
        pass_to_class[pass_index] = cls
    return pass_to_class


def material_index_to_label_map(
    index_map: np.ndarray, pass_to_class: dict[int, int]
) -> np.ndarray:
    """Convert (H, W) pass indices to (H, W) class labels 0..NUM_CLASSES-1."""
    index_flat = index_map.ravel().astype(np.int64)
    out = np.zeros_like(index_flat, dtype=np.uint8)
    for pass_idx, cls in pass_to_class.items():
        out[index_flat == pass_idx] = cls
    return out.reshape(index_map.shape)


def _class_id_for_material_name(mat_name: str) -> int:
    for prefix, class_id in DEFECT_PREFIXES.items():
        if mat_name.startswith(prefix):
            return class_id
    return CLASS_BACKGROUND


def compute_asset_bboxes_from_material_passes(
    index_map: np.ndarray,
    materials_json: dict,
    min_area: int = BBOX_MIN_PIXEL_AREA,
) -> list[dict]:
    """One bbox per defect *material* (Blender pass_index), i.e. per placed plane asset.

    Infinigen assigns a unique pass_index per material datablock at render time; each
    crack/peel plane typically has its own material, so one tight union bbox matches
    one logical object instead of many CC blobs on a merged class mask.

    Each entry: class_id, class_name, pass_index, material_name, x_min, y_min, x_max,
    y_max, width, height, area (pixel count for that pass in this frame).
    """
    by_pass: dict[int, tuple[int, str]] = {}
    for mat_name, info in materials_json.items():
        cls_int = _class_id_for_material_name(mat_name)
        if cls_int == CLASS_BACKGROUND:
            continue
        pass_index = int(info["pass_index"])
        if pass_index not in by_pass:
            by_pass[pass_index] = (cls_int, mat_name)

    boxes: list[dict] = []
    for pass_index in sorted(by_pass.keys()):
        cls_int, mat_name = by_pass[pass_index]
        mask = index_map == pass_index
        ys, xs = np.where(mask)
        if ys.size < min_area:
            continue
        y_min = int(ys.min())
        y_max = int(ys.max())
        x_min = int(xs.min())
        x_max = int(xs.max())
        boxes.append(
            {
                "class_id": cls_int,
                "class_name": CLASS_NAMES.get(cls_int, f"class_{cls_int}"),
                "pass_index": pass_index,
                "material_name": mat_name,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": x_max - x_min + 1,
                "height": y_max - y_min + 1,
                "area": int(ys.size),
            }
        )
    return boxes


def write_bboxes(
    sample_id: str,
    boxes: list[dict],
    image_h: int,
    image_w: int,
    out_bboxes: Path,
    out_bboxes_yolo: Path,
) -> None:
    out_bboxes.mkdir(parents=True, exist_ok=True)
    out_bboxes_yolo.mkdir(parents=True, exist_ok=True)

    payload = {
        "image": f"{sample_id}.png",
        "width": int(image_w),
        "height": int(image_h),
        "boxes": boxes,
    }
    (out_bboxes / f"{sample_id}.json").write_text(json.dumps(payload, indent=2))

    yolo_lines: list[str] = []
    for b in boxes:
        xc = (b["x_min"] + b["x_max"] + 1) / 2.0 / image_w
        yc = (b["y_min"] + b["y_max"] + 1) / 2.0 / image_h
        bw = b["width"] / image_w
        bh = b["height"] / image_h
        # YOLO class ids are 0-indexed; treat defects 1..5 as 0..4.
        yolo_cls = b["class_id"] - 1
        yolo_lines.append(
            f"{yolo_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )
    (out_bboxes_yolo / f"{sample_id}.txt").write_text(
        ("\n".join(yolo_lines) + "\n") if yolo_lines else ""
    )


def discover_scene_frames(root: Path) -> list[tuple[Path, Path, Path, str]]:
    """
    Discover all (image, mask_npy, materials_json, sample_id) under root.
    root can be a single scene dir (e.g. .../room2) or a parent (e.g. .../indoors) containing room1, room2, ...
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return []

    # Case 1: root itself has a frames/ subdir (standard Infinigen layout)
    frames = root / "frames"
    if frames.is_dir():
        return _discover_in_frames(frames, root.name)

    # Case 2: root itself matches the "livingroomXXX" layout: Image/, MaterialSegmentation/, Materials/
    if (
        (root / "Image").is_dir()
        and (root / "MaterialSegmentation").is_dir()
        and (root / "Materials").is_dir()
    ):
        return _discover_in_frames(root, root.name)

    # Case 3: root is a parent containing many scene dirs (bathroom01, livingroom02, ...)
    results = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        f = scene_dir / "frames"
        if f.is_dir():
            results.extend(_discover_in_frames(f, scene_dir.name))
            continue
        if (
            (scene_dir / "Image").is_dir()
            and (scene_dir / "MaterialSegmentation").is_dir()
            and (scene_dir / "Materials").is_dir()
        ):
            results.extend(_discover_in_frames(scene_dir, scene_dir.name))
            continue
        # Case 3b: scene_dir / <rig0_rs0>/Image/... (render output under each rig+resample)
        for leaf in sorted(scene_dir.iterdir()):
            if not leaf.is_dir():
                continue
            if (
                (leaf / "Image").is_dir()
                and (leaf / "MaterialSegmentation").is_dir()
                and (leaf / "Materials").is_dir()
            ):
                results.extend(
                    _discover_in_frames(leaf, f"{scene_dir.name}_{leaf.name}")
                )
    return results


def _discover_in_frames(
    frames_dir: Path, scene_name: str
) -> list[tuple[Path, Path, Path, str]]:
    """Discover triplets inside a frames-like directory.

    Supports both:
      - <scene>/frames/Image/..., MaterialSegmentation/..., Materials/...
      - <scene>/Image/..., MaterialSegmentation/..., Materials/...
    """
    # If frames_dir already is the "frames" folder, its children are Image/, MaterialSegmentation/, Materials/.
    # If frames_dir is the scene dir, it directly contains Image/, MaterialSegmentation/, Materials/.
    if (frames_dir / "Image").is_dir():
        image_dir = frames_dir / "Image"
        seg_dir = frames_dir / "MaterialSegmentation"
        mat_dir = frames_dir / "Materials"
    else:
        image_dir = frames_dir / "frames" / "Image"
        seg_dir = frames_dir / "frames" / "MaterialSegmentation"
        mat_dir = frames_dir / "frames" / "Materials"
    if not image_dir.is_dir() or not seg_dir.is_dir() or not mat_dir.is_dir():
        return []

    # Assume one camera subfolder (camera_0)
    results = []
    for cam_dir in sorted(image_dir.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith("camera_"):
            continue
        subcam = cam_dir.name  # camera_0
        seg_cam = seg_dir / cam_dir.name
        mat_cam = mat_dir / cam_dir.name
        if not seg_cam.is_dir() or not mat_cam.is_dir():
            continue

        # Match Image_<rig>_0_0001_0.png
        pattern = re.compile(r"^Image_(\d+)_0_0001_0\.png$")
        for img_path in sorted(cam_dir.glob("Image_*.png")):
            m = pattern.match(img_path.name)
            if not m:
                continue
            rig = m.group(1)
            stem = f"{rig}_0_0001_0"
            npy_path = seg_cam / f"MaterialSegmentation_{stem}.npy"
            json_path = mat_cam / f"Materials_{stem}.json"
            if not npy_path.is_file() or not json_path.is_file():
                continue
            sample_id = f"{scene_name}_rig{rig}_{subcam}"
            results.append((img_path, npy_path, json_path, sample_id))
    return results


def process_sample(
    image_path: Path,
    npy_path: Path,
    json_path: Path,
    sample_id: str,
    out_images: Path,
    out_masks: Path,
    out_bboxes: Path,
    out_bboxes_yolo: Path,
) -> bool:
    """Load RGB + material data, build label map, save image, mask, and bboxes."""
    try:
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

        with open(json_path) as f:
            materials = json.load(f)
        pass_to_class = build_pass_index_to_class(materials)
        label = material_index_to_label_map(index_map, pass_to_class)

        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)
        imwrite(out_images / f"{sample_id}.png", rgb)
        # Spread grayscale: visible in viewers; distinct from legacy 85/170/255 (see class_names.txt)
        mask_gray = np.clip(
            label.astype(np.uint32) * MASK_GRAY_STEP, 0, 255
        ).astype(np.uint8)
        PILImage.fromarray(mask_gray, mode="L").save(
            str(out_masks / f"{sample_id}.png")
        )

        h, w = label.shape[:2]
        boxes = compute_asset_bboxes_from_material_passes(
            index_map, materials, min_area=BBOX_MIN_PIXEL_AREA
        )
        write_bboxes(sample_id, boxes, h, w, out_bboxes, out_bboxes_yolo)

        return True
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        return False


def collect_complete_sample_ids(out_images: Path, out_masks: Path) -> list[str]:
    """IDs that have both images/<id>.png and masks/<id>.png."""
    if not out_images.is_dir() or not out_masks.is_dir():
        return []
    out: list[str] = []
    for p in sorted(out_images.glob("*.png")):
        sid = p.stem
        if (out_masks / f"{sid}.png").is_file():
            out.append(sid)
    return out


def split_train_val_test(
    ids: list[str], seed: int, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle ids and split ~70% / 15% / 15% (test gets remainder so counts sum to len(ids))."""
    n = len(ids)
    if n == 0:
        return [], [], []
    rng = np.random.default_rng(seed)
    x = list(ids)
    rng.shuffle(x)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val
    train_ids = x[:n_train]
    val_ids = x[n_train : n_train + n_val]
    test_ids = x[n_train + n_val :]
    return train_ids, val_ids, test_ids


DEFAULT_INPUT_FOLDER = Path("/mnt/nvme_storage/dataset/all_frames")
DEFAULT_OUTPUT_FOLDER = Path("/mnt/nvme_storage/dataset/defect_segmentation_dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Build defect segmentation dataset from Infinigen frames (RGB + MaterialSegmentation + Materials JSON)."
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        type=Path,
        default=DEFAULT_INPUT_FOLDER,
        help=(
            "Root directory: single scene, or parent of many scenes "
            "(e.g. all_frames with bathroom01/rig0_rs0/Image/...). "
            f"Default: {DEFAULT_INPUT_FOLDER}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help=(
            "Output directory for images/, masks/, splits/. "
            f"Default: {DEFAULT_OUTPUT_FOLDER}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-write images and masks even if they already exist.",
    )
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Only refresh splits/class_names from existing images/ and masks/ (no input scan).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (e.g. for logs).",
    )
    args = parser.parse_args()

    input_root = args.input_folder.resolve()
    output_dir = args.output_folder.resolve()
    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_bboxes = output_dir / "bboxes"
    out_bboxes_yolo = output_dir / "bboxes_yolo"

    samples: list[tuple[Path, Path, Path, str]] = []
    if not args.splits_only:
        samples = discover_scene_frames(input_root)
        if not samples:
            existing = collect_complete_sample_ids(out_images, out_masks)
            if existing:
                print(
                    f"No frame triplets under {input_root}; "
                    f"refreshing splits from {len(existing)} existing image+mask pairs."
                )
            else:
                print(f"No frame triplets found under {input_root} and no existing outputs.")
                return 1
        else:
            print(f"Found {len(samples)} samples (input: {input_root}).")
            skipped = 0
            success = 0
            iterator = (
                samples
                if args.no_progress
                else tqdm(samples, desc="Processing samples", unit="sample")
            )
            for img_path, npy_path, json_path, sample_id in iterator:
                img_out = out_images / f"{sample_id}.png"
                mask_out = out_masks / f"{sample_id}.png"
                bbox_out = out_bboxes / f"{sample_id}.json"
                yolo_out = out_bboxes_yolo / f"{sample_id}.txt"
                if (
                    not args.force
                    and img_out.is_file()
                    and mask_out.is_file()
                    and bbox_out.is_file()
                    and yolo_out.is_file()
                ):
                    skipped += 1
                    continue
                if process_sample(
                    img_path,
                    npy_path,
                    json_path,
                    sample_id,
                    out_images,
                    out_masks,
                    out_bboxes,
                    out_bboxes_yolo,
                ):
                    success += 1
            print(
                f"Wrote {success} new/updated samples, skipped {skipped} already present "
                f"(of {len(samples)} from input) -> {output_dir}"
            )
    else:
        print("--splits-only: skipping input processing.")

    ids = collect_complete_sample_ids(out_images, out_masks)
    if not ids:
        print(f"No complete image+mask pairs under {output_dir}")
        return 1

    train_ids, val_ids, test_ids = split_train_val_test(ids, args.seed)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(val_ids) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(test_ids) + "\n")
    print(
        f"Wrote splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test "
        f"-> {splits_dir}"
    )

    # Class legend for training
    legend = output_dir / "class_names.txt"
    half = MASK_GRAY_STEP // 2
    legend.write_text(
        "0 background\n"
        "1 crack (CrackMaterial_*)\n"
        "2 paint_peel (PaintPeelMaterial_*)\n"
        "3 spalling (SpallingMaterial_*)\n"
        "4 paint_bubble (BubbleMaterial_*)\n"
        "5 exposed_wiring (OpenWiringMaterial_*)\n"
        f"# Mask PNG: gray level = class_id * {MASK_GRAY_STEP} (class 0..{NUM_CLASSES - 1})\n"
        f"# Decode: label = ((mask + {half}) // {MASK_GRAY_STEP}).clip(0, {NUM_CLASSES - 1})\n"
        "# Legacy 4-class masks only use {0,85,170,255}: label = (mask // 85).clip(0,3)\n"
        "# 2D bboxes: per-image JSON — one box per defect material pass_index (placed asset);\n"
        "#   boxes[{class_id, class_name, pass_index, material_name, x_min, y_min, x_max,\n"
        "#   y_max, width, height, area}] (pixel coords, inclusive).\n"
        "# YOLO labels: bboxes_yolo/<id>.txt lines '<cls> <xc_n> <yc_n> <w_n> <h_n>', defects 1..5\n"
        "# are remapped to YOLO class ids 0..4 (background is not an object class).\n"
    )
    print(f"Class legend: {legend}")

    write_coco_aggregate(output_dir, out_bboxes, ids)

    return 0


def write_coco_aggregate(
    output_dir: Path, out_bboxes: Path, ids: list[str]
) -> None:
    """Combine per-image bbox JSONs into a single COCO-format annotations.json."""
    if not out_bboxes.is_dir():
        return

    images = []
    annotations = []
    ann_id = 1
    for image_id, sid in enumerate(ids, start=1):
        bbox_file = out_bboxes / f"{sid}.json"
        if not bbox_file.is_file():
            continue
        try:
            data = json.loads(bbox_file.read_text())
        except Exception:
            continue
        images.append(
            {
                "id": image_id,
                "file_name": f"{sid}.png",
                "width": data.get("width", 0),
                "height": data.get("height", 0),
            }
        )
        for b in data.get("boxes", []):
            w = b["x_max"] - b["x_min"] + 1
            h = b["y_max"] - b["y_min"] + 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(b["class_id"]),
                    "bbox": [int(b["x_min"]), int(b["y_min"]), int(w), int(h)],
                    "area": int(b.get("area", w * h)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [
        {"id": cid, "name": CLASS_NAMES[cid]}
        for cid in sorted(DEFECT_PREFIXES.values())
    ]
    coco = {
        "info": {
            "description": "Infinigen defect 2D bboxes (one box per material pass_index / asset)"
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_path = output_dir / "annotations_coco.json"
    coco_path.write_text(json.dumps(coco))
    print(
        f"Wrote COCO aggregate: {len(images)} images, {len(annotations)} boxes -> {coco_path}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

# Usage:
#   # Defaults: -i /mnt/nvme_storage/dataset/all_frames, -o /mnt/nvme_storage/dataset/defect_segmentation_dataset
#   python prepare_defect_annotated_dataset.py
#
#   python prepare_defect_annotated_dataset.py -i /path/to/all_frames -o /path/to/out
#
# Output: <output-folder>/images/*.png, masks/*.png, splits/train.txt, val.txt, test.txt, class_names.txt
# Re-run: skips existing pairs unless --force; always rewrites splits from all complete pairs.
# Splits only: python prepare_defect_annotated_dataset.py --splits-only -o <out>
