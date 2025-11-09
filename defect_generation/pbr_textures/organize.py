import os
import shutil
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
SOURCE_FOLDER = r"C:\Users\shive\OneDrive\Desktop\cuhk_research_tools\defect_detection_dataset\defect_generation\pbr_textures"
OUTPUT_BASE_FOLDER = SOURCE_FOLDER  # Organize in the same folder
DRY_RUN = False  # Set to True to preview without moving files

# Common PBR texture prefixes
TEXTURE_TYPES = ['height', 'mask', 'normal', 'opacity', 'rough', 'upscale', 'displacement', 'basecolor', 'albedo', 'metallic', 'ao']

def extract_base_number(filename):
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

def get_texture_type(filename):
    """
    Extract texture type from filename (e.g., 'opacity', 'rough', 'normal')
    """
    for tex_type in TEXTURE_TYPES:
        if filename.lower().startswith(tex_type + '_'):
            return tex_type
    return 'unknown'

def generate_unique_id():
    """
    Generate a unique identifier based on timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def organize_textures(source_folder, dry_run=False):
    """
    Organize texture files into folders by their base number
    """
    source_path = Path(source_folder)
    
    if not source_path.exists():
        print(f"✗ Source folder does not exist: {source_folder}")
        return
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.exr']
    all_files = []
    for ext in image_extensions:
        all_files.extend(source_path.glob(f"*{ext}"))
    
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
        
        # Create folder name with unique ID
        folder_name = f"texture_{base_num}_{unique_id}"
        folder_path = source_path / folder_name
        
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
    manifest_path = source_path / f"organization_manifest_{unique_id}.txt"
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

if __name__ == "__main__":
    # Set DRY_RUN = True to preview first
    organize_textures(SOURCE_FOLDER, dry_run=DRY_RUN)