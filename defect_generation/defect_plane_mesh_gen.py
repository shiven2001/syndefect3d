import bpy
import os
import math
from pathlib import Path

# Configuration
# Use this path on Linux:
PBR_TEXTURES_FOLDER = r"C:\Users\shive\OneDrive\Documents\GitHub\syndefect3d\defect_generation\pbr_textures"
# Or use this on Windows:
# PBR_TEXTURES_FOLDER = r"C:\Users\shive\OneDrive\Desktop\cuhk_research_tools\defect_detection_dataset\defect_generation\pbr_textures"

FOLDER_SUFFIX = "20251114_103900"  # Set to None to process all folders, or specify like "20251109_165811"
OUTPUT_FOLDER = "glb_exports"  # Where to save GLB files (relative to script folder)
PLANE_SCALE = (0.8, 0.6, 1.0)  # Scale for wall decoration
SOLIDIFY_THICKNESS = 0.005  # 5mm - very thin

# Get script directory for output path (use parent of PBR textures folder)
SCRIPT_DIR = Path(PBR_TEXTURES_FOLDER).parent

# Texture map suffixes to look for
TEXTURE_MAP_TYPES = {
    "albedo": ["upscale", "basecolor", "albedo", "diffuse"],
    "roughness": ["rough", "roughness"],
    "normal": ["normal"],
    "mask": ["mask"],
    "opacity": ["opacity", "alpha"],
    "height": ["height", "displacement", "bump"],
}


def find_texture_folders(base_path, suffix_filter=None):
    """Find all texture_* folders in the base path, optionally filtered by suffix"""
    folders = []
    for item in Path(base_path).iterdir():
        if item.is_dir() and item.name.startswith("texture_"):
            # Apply suffix filter if specified
            if suffix_filter is None or item.name.endswith(suffix_filter):
                folders.append(item)
    return sorted(folders)


def find_texture_map(folder_path, map_types):
    """Find a texture map in the folder matching any of the map types"""
    for map_type in map_types:
        for file in folder_path.iterdir():
            if file.is_file() and map_type in file.stem.lower():
                return str(file)
    return None


def create_pbr_material(name, textures):
    """Create a PBR material with the given textures"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create shader nodes
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (400, 300)

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (700, 300)

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    y_offset = 600

    # Albedo/Base Color
    if textures.get("albedo"):
        tex_albedo = nodes.new(type="ShaderNodeTexImage")
        tex_albedo.location = (-400, y_offset)
        tex_albedo.label = "Albedo"
        tex_albedo.image = bpy.data.images.load(textures["albedo"])
        links.new(tex_albedo.outputs["Color"], bsdf.inputs["Base Color"])
        print(f"  ✓ Loaded Albedo")
        y_offset -= 300

    # Roughness
    if textures.get("roughness"):
        tex_rough = nodes.new(type="ShaderNodeTexImage")
        tex_rough.location = (-400, y_offset)
        tex_rough.label = "Roughness"
        tex_rough.image = bpy.data.images.load(textures["roughness"])
        tex_rough.image.colorspace_settings.name = "Non-Color"
        links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])
        print(f"  ✓ Loaded Roughness")
        y_offset -= 300

    # Normal Map
    if textures.get("normal"):
        tex_normal = nodes.new(type="ShaderNodeTexImage")
        tex_normal.location = (-400, y_offset)
        tex_normal.label = "Normal"
        tex_normal.image = bpy.data.images.load(textures["normal"])
        tex_normal.image.colorspace_settings.name = "Non-Color"

        normal_map = nodes.new(type="ShaderNodeNormalMap")
        normal_map.location = (100, y_offset)

        links.new(tex_normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        print(f"  ✓ Loaded Normal")
        y_offset -= 300

    # Opacity/Alpha - Use mask texture for proper transparency
    mask_texture = textures.get("mask") or textures.get("opacity")
    if mask_texture:
        tex_opacity = nodes.new(type="ShaderNodeTexImage")
        tex_opacity.location = (-700, y_offset)
        tex_opacity.label = "Opacity/Mask"
        tex_opacity.image = bpy.data.images.load(mask_texture)
        tex_opacity.image.colorspace_settings.name = "Non-Color"

        # Invert the mask (if mask is black=defect, white=transparent, we need to flip it)
        invert_node = nodes.new(type="ShaderNodeInvert")
        invert_node.location = (-400, y_offset)
        links.new(tex_opacity.outputs["Color"], invert_node.inputs["Color"])

        # Connect inverted mask to Alpha
        links.new(invert_node.outputs["Color"], bsdf.inputs["Alpha"])

        # Set material for proper transparency export
        mat.blend_method = "BLEND"
        mat.shadow_method = "CLIP"
        mat.use_backface_culling = False

        print(f"  ✓ Loaded Opacity/Mask (inverted)")
        y_offset -= 300

    # Height/Displacement
    if textures.get("height"):
        tex_height = nodes.new(type="ShaderNodeTexImage")
        tex_height.location = (-400, y_offset)
        tex_height.label = "Height"
        tex_height.image = bpy.data.images.load(textures["height"])
        tex_height.image.colorspace_settings.name = "Non-Color"

        # Use as bump map
        bump = nodes.new(type="ShaderNodeBump")
        bump.location = (100, y_offset)
        bump.inputs["Strength"].default_value = 0.5

        links.new(tex_height.outputs["Color"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        print(f"  ✓ Loaded Height/Displacement")

    return mat


def create_defect_plane(name, material):
    """Create a defect plane with material"""
    # Create plane
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.active_object
    plane.name = name

    # Scale to reasonable size
    plane.scale = PLANE_SCALE

    # Rotate so back face will be against wall (minimum X direction)
    plane.rotation_euler = (0, math.pi / 2, 0)  # 90° around Y-axis

    # Apply transforms
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Add very thin thickness
    bpy.ops.object.modifier_add(type="SOLIDIFY")
    plane.modifiers["Solidify"].thickness = SOLIDIFY_THICKNESS
    bpy.ops.object.modifier_apply(modifier="Solidify")

    # Triangulate
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")

    # Apply material
    if plane.data.materials:
        plane.data.materials[0] = material
    else:
        plane.data.materials.append(material)

    return plane


def export_to_glb(obj, output_path):
    """Export object to GLB file"""
    # Select only this object
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    print(f"  → Exporting to: {output_path}")

    # Export as GLB
    try:
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            use_selection=True,
            export_format="GLB",
            export_texcoords=True,
            export_normals=True,
            export_materials="EXPORT",
            export_cameras=False,
            export_lights=False,
        )
        # Check if file was created
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            print(f"  ✓ File saved successfully ({file_size} bytes)")
        else:
            print(f"  ✗ File was not created!")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")


def process_all_textures():
    """Main function to process all texture folders and export GLB files"""
    base_path = Path(PBR_TEXTURES_FOLDER)

    print(f"\n📁 Texture folder: {PBR_TEXTURES_FOLDER}")
    print(f"📁 Script directory: {SCRIPT_DIR}")

    if not base_path.exists():
        print(f"✗ Folder not found: {PBR_TEXTURES_FOLDER}")
        return

    # Create output folder in script directory
    output_path = SCRIPT_DIR / OUTPUT_FOLDER
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output folder: {output_path.absolute()}")
    print(f"   (This is where GLB files will be saved)")

    # Find all texture folders with suffix filter
    texture_folders = find_texture_folders(base_path, FOLDER_SUFFIX)

    if not texture_folders:
        if FOLDER_SUFFIX:
            print(f"✗ No texture folders found with suffix '{FOLDER_SUFFIX}'!")
        else:
            print("✗ No texture folders found!")
        return

    print(f"\n✓ Found {len(texture_folders)} texture sets")
    if FOLDER_SUFFIX:
        print(f"✓ Filtered by suffix: {FOLDER_SUFFIX}")
    print("\n" + "=" * 70)

    created_count = 0

    for idx, folder in enumerate(texture_folders):
        print(f"\n[{idx + 1}/{len(texture_folders)}] Processing: {folder.name}")

        # Clear existing mesh objects
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        # Find texture maps
        textures = {}
        for map_name, map_types in TEXTURE_MAP_TYPES.items():
            texture_path = find_texture_map(folder, map_types)
            if texture_path:
                textures[map_name] = texture_path

        if not textures:
            print(f"  ✗ No textures found in {folder.name}")
            continue

        # Create material
        mat_name = f"Material_{folder.name}"
        material = create_pbr_material(mat_name, textures)

        # Create plane
        plane_name = f"defect_{folder.name}"
        plane = create_defect_plane(plane_name, material)

        print(f"  ✓ Created defect plane")

        # Export to GLB
        glb_filename = f"{folder.name}.glb"
        glb_path = str(output_path / glb_filename)
        export_to_glb(plane, glb_path)
        created_count += 1

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total texture sets found: {len(texture_folders)}")
    print(f"GLB files created: {created_count}")
    print(f"Output location: {output_path.absolute()}")
    if FOLDER_SUFFIX:
        print(f"Suffix filter: {FOLDER_SUFFIX}")

    # List the GLB files that were created
    glb_files = list(output_path.glob("*.glb"))
    if glb_files:
        print(f"\n📦 GLB files in output folder:")
        for glb_file in sorted(glb_files):
            size_kb = glb_file.stat().st_size / 1024
            print(f"   • {glb_file.name} ({size_kb:.1f} KB)")
    else:
        print(f"\n⚠️  Warning: No GLB files found in {output_path.absolute()}")

    print("=" * 70)


# Run the script
import sys

try:
    print("=" * 70, file=sys.stderr)
    print("STARTING DEFECT PLANE GENERATION", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.stderr.flush()

    print("=" * 70)
    print("STARTING DEFECT PLANE GENERATION")
    print("=" * 70)

    process_all_textures()

    print("\n✓✓✓ SCRIPT COMPLETED SUCCESSFULLY ✓✓✓", file=sys.stderr)
    sys.stderr.flush()
except Exception as e:
    print("\n" + "=" * 70, file=sys.stderr)
    print("ERROR OCCURRED:", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc(file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.stderr.flush()
