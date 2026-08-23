#!/usr/bin/env python3
"""
Export masks (and optionally bbox JSON + YOLO) for **one** sample id, using the same logic as
prepare_defect_annotated_dataset.py (no full-dataset scan beyond discovery).

Typical use after rendering under all_frames:
  python tools/prepare_defect_single_sample.py \\
    -i /mnt/nvme_storage/syndefect3d_dataset_v2/all_frames \\
    --sample-id bathroom08_rig18_rs1_rig18_camera_0 \\
    -o /tmp/bathroom08_one_sample

Writes under -o (default: segmentation only):
  images/<id>.png  masks/<id>.png  class_names.txt  splits/sample.txt

With --with-bboxes also:
  bboxes/<id>.json  bboxes_yolo/<id>.txt  annotations_coco.json
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
    parser.add_argument(
        "--with-bboxes",
        action="store_true",
        help="Also write bboxes/, bboxes_yolo/, and annotations_coco.json (same as full preparer).",
    )
    parser.add_argument(
        "--realism-postprocess",
        action="store_true",
        help="Desaturate, brighten, and add grain to exported RGB.",
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

    required = [img_out, mask_out]
    if args.with_bboxes:
        required.extend([bbox_out, yolo_out])
    if not args.force and all(p.is_file() for p in required):
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
        with_bboxes=args.with_bboxes,
        realism_postprocess=args.realism_postprocess,
    )
    if not ok:
        return 1

    prep.write_class_names_legend(out)
    if args.with_bboxes:
        prep.write_coco_aggregate(out, out_bboxes, [sid])
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits" / "sample.txt").write_text(sid + "\n")

    print(f"OK → {out}")
    print(f"  RGB:   {img_path}")
    print(f"  Seg:   {npy_path}")
    print(f"  Mat:   {json_path}")
    wrote = [img_out.name, mask_out.name]
    if args.with_bboxes:
        wrote.extend([bbox_out.name, yolo_out.name])
    print(f"  Wrote: {', '.join(wrote)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
