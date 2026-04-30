#!/usr/bin/env python3
"""
Export masks + bbox JSON + YOLO labels for **one** sample id, using the same logic as
prepare_defect_annotated_dataset.py (no full-dataset scan beyond discovery).

Typical use after rendering under all_frames:
  python tools/prepare_defect_single_sample.py \\
    -i /mnt/nvme_storage/syndefect3d_dataset_v2/all_frames \\
    --sample-id bathroom08_rig18_rs1_rig18_camera_0 \\
    -o /tmp/bathroom08_one_sample

Writes under -o:
  images/<id>.png  masks/<id>.png  bboxes/<id>.json  bboxes_yolo/<id>.txt
  class_names.txt (same file as full preparer: prepare_defect_annotated_dataset.write_class_names_legend)
  annotations_coco.json (single image)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import prepare_defect_annotated_dataset as prep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input-folder",
        type=Path,
        required=True,
        help="Root for discover_scene_frames (e.g. all_frames containing bathroom08/rig18_rs1/...).",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        required=True,
        help="Exact sample id, e.g. bathroom08_rig18_rs1_rig18_camera_0",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=Path,
        default=Path("single_defect_sample"),
        help="Output directory (default: ./single_defect_sample)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    args = parser.parse_args()

    input_root = args.input_folder.resolve()
    out = args.output_folder.resolve()
    sample_id = args.sample_id.strip()

    samples = prep.discover_scene_frames(input_root)
    if not samples:
        print(f"No frame triplets under {input_root}", file=sys.stderr)
        return 1

    chosen = None
    for img_path, npy_path, json_path, sid in samples:
        if sid == sample_id:
            chosen = (img_path, npy_path, json_path, sid)
            break

    if chosen is None:
        print(
            f"Sample id {sample_id!r} not found under {input_root} "
            f"({len(samples)} samples discovered). First few ids:",
            file=sys.stderr,
        )
        for _, _, _, sid in samples[:15]:
            print(f"  {sid}", file=sys.stderr)
        if len(samples) > 15:
            print(f"  ... and {len(samples) - 15} more", file=sys.stderr)
        return 1

    img_path, npy_path, json_path, sid = chosen
    out_images = out / "images"
    out_masks = out / "masks"
    out_bboxes = out / "bboxes"
    out_bboxes_yolo = out / "bboxes_yolo"

    img_out = out_images / f"{sid}.png"
    mask_out = out_masks / f"{sid}.png"
    bbox_out = out_bboxes / f"{sid}.json"
    yolo_out = out_bboxes_yolo / f"{sid}.txt"

    if not args.force and all(
        p.is_file() for p in (img_out, mask_out, bbox_out, yolo_out)
    ):
        print(
            f"Outputs already exist under {out}; use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    ok = prep.process_sample(
        img_path,
        npy_path,
        json_path,
        sid,
        out_images,
        out_masks,
        out_bboxes,
        out_bboxes_yolo,
    )
    if not ok:
        return 1

    prep.write_class_names_legend(out)
    prep.write_coco_aggregate(out, out_bboxes, [sid])
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits" / "sample.txt").write_text(sid + "\n")

    print(f"OK → {out}")
    print(f"  RGB:   {img_path}")
    print(f"  Seg:   {npy_path}")
    print(f"  Mat:   {json_path}")
    print(f"  Wrote: {img_out.name}, {mask_out.name}, {bbox_out.name}, {yolo_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
