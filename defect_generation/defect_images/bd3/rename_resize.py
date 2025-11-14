from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
TMP_PREFIX = "__ren_tmp__"


def resize_image(
    input_path: Path,
    size: Tuple[int, int],
    maintain_aspect: bool,
) -> None:
    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)
        if maintain_aspect:
            img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        else:
            img = img.resize(size, Image.Resampling.LANCZOS)

        if img.mode == "RGBA" and input_path.suffix.lower() in {'.jpg', '.jpeg'}:
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background

        save_kwargs = {"optimize": True}
        if input_path.suffix.lower() in {'.jpg', '.jpeg', '.webp'}:
            save_kwargs["quality"] = 95

        img.save(input_path, **save_kwargs)


def gather_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name,
    )


def rename_sequential(
    image_paths: List[Path],
    prefix: str,
    pad: int,
) -> List[Path]:
    temp_paths: List[Path] = []
    for idx, path in enumerate(image_paths, 1):
        tmp_name = f"{TMP_PREFIX}{idx:06d}{path.suffix.lower()}"
        tmp_path = path.with_name(tmp_name)
        path.rename(tmp_path)
        temp_paths.append(tmp_path)

    final_paths: List[Path] = []
    for idx, tmp_path in enumerate(temp_paths, 1):
        new_name = f"{prefix}_{idx:0{pad}d}{tmp_path.suffix.lower()}"
        new_path = tmp_path.with_name(new_name)
        tmp_path.rename(new_path)
        final_paths.append(new_path)

    return final_paths


def process_folder(
    folder: Path,
    prefix: str,
    pad: int,
    size: Tuple[int, int],
    maintain_aspect: bool,
) -> None:
    folder = folder.expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = gather_images(folder)
    if not images:
        print(f"No images found in {folder}")
        return

    print(f"Found {len(images)} images in {folder}")
    renamed = rename_sequential(images, prefix=prefix, pad=pad)
    print("✓ Renamed files sequentially")

    for path in renamed:
        resize_image(path, size=size, maintain_aspect=maintain_aspect)
    print(f"✓ Resized all images to {size[0]}x{size[1]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Rename BD3 images sequentially and resize them."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=default_folder,
        help=f"Target folder (default: {default_folder})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="IMG",
        help="Prefix for renamed images (default: IMG)",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=4,
        help="Number of digits for numbering (default: 4 -> IMG_0001)",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="Resize target size (default: 1024 1024)",
    )
    parser.add_argument(
        "--maintain-aspect",
        action="store_true",
        help="Maintain aspect ratio with center crop (default: stretch).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    process_folder(
        folder=args.folder,
        prefix=args.prefix,
        pad=args.pad,
        size=tuple(args.size),
        maintain_aspect=args.maintain_aspect,
    )


if __name__ == "__main__":
    main()
