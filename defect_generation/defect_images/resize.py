import os
from PIL import Image
from pathlib import Path

# Configuration
INPUT_FOLDER = r"C:\Users\shive\OneDrive\Desktop\cuhk_research_tools\defect_detection_dataset\defect_generation\defect_images"
OUTPUT_FOLDER = INPUT_FOLDER  # Same as input (not used when overwriting)
TARGET_SIZE = (1024, 1024)
OVERWRITE_ORIGINAL = True  # Set to True to replace original files

# Supported image formats
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

def resize_image(input_path, output_path, size=(1024, 1024), maintain_aspect=False):
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
            # Convert RGBA to RGB if saving as JPEG
            if img.mode == 'RGBA' and output_path.lower().endswith(('.jpg', '.jpeg')):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                img = background
            
            if maintain_aspect:
                # Resize maintaining aspect ratio and crop to square
                img.thumbnail((size[0] * 2, size[1] * 2), Image.Resampling.LANCZOS)
                
                # Create a new square image and paste the resized image centered
                new_img = Image.new('RGB', size, (255, 255, 255))
                offset = ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2)
                new_img.paste(img, offset)
                img = new_img
            else:
                # Direct resize (may distort if aspect ratio changes)
                img = img.resize(size, Image.Resampling.LANCZOS)
            
            # Save the resized image
            img.save(output_path, quality=95, optimize=True)
            return True
    except Exception as e:
        print(f"✗ Error processing {input_path}: {e}")
        return False

def resize_all_images(input_folder, output_folder, size=(1024, 1024), overwrite=False):
    """
    Resize all images in a folder and its subfolders.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"✗ Input folder does not exist: {input_folder}")
        return
    
    # Create output folder if it doesn't exist and we're not overwriting
    if not overwrite:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    total_files = len(image_files)
    print(f"Found {total_files} image files to resize\n")
    print(f"Target size: {size[0]}x{size[1]}")
    print(f"{'Overwriting original files' if overwrite else f'Saving to: {output_folder}'}\n")
    
    if total_files == 0:
        print("No images found!")
        return
    
    # Ask for confirmation
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
        
        if resize_image(str(img_file), str(out_file), size):
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

if __name__ == "__main__":
    # Option 1: Resize to new folder (recommended - preserves originals)
    resize_all_images(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_SIZE, overwrite=OVERWRITE_ORIGINAL)
    
    # Option 2: Resize specific folder without subfolders
    # resize_all_images(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_SIZE, overwrite=False)