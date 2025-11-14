from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Common PBR texture prefixes
TEXTURE_TYPES = [
    'height',
    'mask',
    'normal',
    'opacity',
    'rough',
    'upscale',
    'displacement',
    'basecolor',
    'albedo',
    'metallic',
    'ao',
]
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr']

def extract_base_number(filename: str) -> str | None:
    """
    Extract the base number from filenames like 'opacity_00001_.png'
    Returns: base number string (e.g., '00001') or None
    """
    # Pattern to match texture_XXXXX_ format
    pattern = r'(?:' + '|'.join(TEXTURE_TYPES) + r')_(\d+)_'
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def get_texture_type(filename: str) -> str:
    """
    Extract texture type from filename (e.g., 'opacity', 'rough', 'normal')
    """
    for tex_type in TEXTURE_TYPES:
        if filename.lower().startswith(tex_type + '_'):
            return tex_type
    return 'unknown'

def generate_unique_id() -> str:
    """
    Generate a unique identifier based on timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def organize_textures(
    source_folder: Path,
    output_base_folder: Path | None = None,
    dry_run: bool = False,
    recursive: bool = True,
) -> None:
    """
    Organize texture files into folders by their base number
    """
    source_path = Path(source_folder).expanduser().resolve()
    output_base = (
        Path(output_base_folder).expanduser().resolve()
        if output_base_folder
        else source_path
    )
    
    if not source_path.exists():
        print(f"✗ Source folder does not exist: {source_folder}")
        return
    
    all_files = list(_find_image_files(source_path, recursive))
    
    # Group files by base number
    texture_groups = defaultdict(list)
    
    for file_path in all_files:
        base_num = extract_base_number(file_path.name)
        if base_num:
            texture_groups[base_num].append(file_path)
    
    if not texture_groups:
        print("✗ No texture files found matching the pattern!")
        return
    
    print(f"Found {len(texture_groups)} texture sets")
    print(f"Total files: {sum(len(files) for files in texture_groups.values())}\n")
    
    # Display preview
    print("="*70)
    print("PREVIEW OF ORGANIZATION")
    print("="*70)
    for base_num in sorted(texture_groups.keys()):
        files = texture_groups[base_num]
        print(f"\nSet '{base_num}': {len(files)} files")
        for f in files:
            print(f"  - {f.name}")
    
    print("\n" + "="*70)
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be moved")
        return
    
    # Ask for confirmation
    response = input("\nProceed with organizing? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Operation cancelled.")
        return
    
    # Generate unique ID for this batch
    unique_id = generate_unique_id()
    print(f"\nUnique ID for this batch: {unique_id}\n")
    
    print("="*70)
    print("ORGANIZING FILES")
    print("="*70 + "\n")
    
    organized_count = 0
    
    for base_num in sorted(texture_groups.keys()):
        files = texture_groups[base_num]
        
        folder_name = f"texture_{base_num}_{unique_id}"
        folder_path = output_base / folder_name
        
        try:
            # Create folder
            folder_path.mkdir(exist_ok=True)
            print(f"📁 Created folder: {folder_name}")
            
            # Move and rename files
            for file_path in files:
                # Get texture type
                tex_type = get_texture_type(file_path.name)
                
                # Create new filename with unique ID
                # Format: texturetype_basenumber_uniqueid.ext
                new_filename = f"{tex_type}_{base_num}_{unique_id}{file_path.suffix}"
                new_path = folder_path / new_filename
                
                # Move file
                shutil.move(str(file_path), str(new_path))
                print(f"  ✓ Moved: {file_path.name} → {folder_name}/{new_filename}")
                organized_count += 1
            
            print()
            
        except Exception as e:
            print(f"  ✗ Error organizing set {base_num}: {e}\n")
    
    # Summary
    print("="*70)
    print("ORGANIZATION COMPLETE")
    print("="*70)
    print(f"Texture sets organized: {len(texture_groups)}")
    print(f"Total files moved: {organized_count}")
    print(f"Unique batch ID: {unique_id}")
    print("="*70)
    
    # Create a manifest file
    manifest_path = output_base / f"organization_manifest_{unique_id}.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"Texture Organization Manifest\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Unique Batch ID: {unique_id}\n")
        f.write(f"Total Sets: {len(texture_groups)}\n")
        f.write(f"Total Files: {organized_count}\n\n")
        f.write("="*70 + "\n")
        
        for base_num in sorted(texture_groups.keys()):
            folder_name = f"texture_{base_num}_{unique_id}"
            f.write(f"\n{folder_name}:\n")
            for file_path in texture_groups[base_num]:
                tex_type = get_texture_type(file_path.name)
                new_filename = f"{tex_type}_{base_num}_{unique_id}{file_path.suffix}"
                f.write(f"  - {new_filename}\n")
    
    print(f"\n📄 Manifest saved: {manifest_path.name}")

def _find_image_files(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in folder.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group PBR texture maps into numbered folders."
    )
    default_source = Path(__file__).resolve().parent

    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help="Folder that contains the textures (default: folder containing this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder to place organized sets (default: same as source)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only look at files directly inside --source (skip subfolders).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview organization without moving files.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    organize_textures(
        source_folder=args.source,
        output_base_folder=args.output,
        dry_run=args.dry_run,
        recursive=not args.no_recursive,
    )