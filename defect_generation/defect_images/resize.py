from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageOps

# Supported image formats (lowercase for easy comparison)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def resize_image(
    input_path: str | Path,
    output_path: str | Path,
    size: Tuple[int, int] = (1024, 1024),
    maintain_aspect: bool = False,
) -> bool:
    """
    Resize an image to the specified size.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        size: Target size tuple (width, height)
        maintain_aspect: If True, maintains aspect ratio with padding/cropping
    """
    try:
        with Image.open(input_path) as img:
            img = ImageOps.exif_transpose(img)

            if maintain_aspect:
                img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            else:
                img = img.resize(size, Image.Resampling.LANCZOS)

            # Convert RGBA to RGB if saving as JPEG
            destination = str(output_path).lower()
            if img.mode == 'RGBA' and destination.endswith(('.jpg', '.jpeg')):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background

            save_kwargs = {"optimize": True}
            if destination.endswith(('.jpg', '.jpeg', '.webp')):
                save_kwargs["quality"] = 95

            img.save(output_path, **save_kwargs)
            return True
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def resize_all_images(
    input_folder: Path,
    output_folder: Path | None,
    size: Tuple[int, int] = (1024, 1024),
    overwrite: bool = False,
    maintain_aspect: bool = False,
    skip_confirmation: bool = False,
) -> None:
    """
    Resize all images in a folder and its subfolders.
    """
    input_path = Path(input_folder).expanduser().resolve()
    output_path = (
        input_path if overwrite else Path(output_folder or input_path / "resized").resolve()
    )

    if not input_path.exists():
        print(f"✗ Input folder does not exist: {input_path}")
        return

    if not overwrite:
        output_path.mkdir(parents=True, exist_ok=True)

    image_files = _gather_images(input_path)
    total_files = len(image_files)
    print(f"Found {total_files} image files to resize\n")
    print(f"Target size: {size[0]}x{size[1]}")
    print(f"{'Overwriting original files' if overwrite else f'Saving to: {output_folder}'}\n")
    
    if total_files == 0:
        print("No images found!")
        return
    
    if not skip_confirmation:
        response = input("Proceed with resizing? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Operation cancelled.")
            return
    
    print("\n" + "="*60)
    print("Starting image resize...")
    print("="*60 + "\n")
    
    success_count = 0
    failed_count = 0
    
    for i, img_file in enumerate(image_files, 1):
        # Calculate relative path to maintain folder structure
        rel_path = img_file.relative_to(input_path)
        
        if overwrite:
            out_file = img_file
        else:
            out_file = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i}/{total_files}] Processing: {rel_path}")
        
        if resize_image(str(img_file), str(out_file), size, maintain_aspect=maintain_aspect):
            success_count += 1
            print(f"  ✓ Resized successfully")
        else:
            failed_count += 1
        
        # Progress indicator
        if i % 10 == 0:
            print(f"\nProgress: {i}/{total_files} ({i/total_files*100:.1f}%)\n")
    
    # Summary
    print("\n" + "="*60)
    print("RESIZE COMPLETE")
    print("="*60)
    print(f"Total files: {total_files}")
    print(f"Successfully resized: {success_count}")
    print(f"Failed: {failed_count}")
    print("="*60)

def _gather_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch resize images to a fixed size.")
    default_input = Path(__file__).resolve().parent

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Folder containing images (default: folder containing this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder to write resized images (default: <input>/resized when not overwriting)",
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="Target size as WIDTH HEIGHT (default: 1024 1024)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original images in-place instead of writing to an output folder.",
    )
    parser.add_argument(
        "--maintain-aspect",
        action="store_true",
        help="Maintain aspect ratio and crop/letterbox to the requested size.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    resize_all_images(
        input_folder=args.input,
        output_folder=args.output,
        size=tuple(args.size),
        overwrite=args.overwrite,
        maintain_aspect=args.maintain_aspect,
        skip_confirmation=args.yes,
    )