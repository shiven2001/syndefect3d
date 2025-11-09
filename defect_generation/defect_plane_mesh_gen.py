import bpy
import os
from pathlib import Path

# Configuration
PBR_TEXTURES_FOLDER = r"C:\Users\shive\OneDrive\Desktop\cuhk_research_tools\defect_detection_dataset\defect_generation\pbr_textures"
FOLDER_SUFFIX = "20251109_165811"  # Set to None to process all folders, or specify like "20251109_165811"
PLANE_SIZE = 2.0
GRID_SPACING = 3.0  # Space between planes
PLANES_PER_ROW = 5  # How many planes per row in the grid

# Texture map suffixes to look for
TEXTURE_MAP_TYPES = {
    'albedo': ['upscale', 'basecolor', 'albedo', 'diffuse'],
    'roughness': ['rough', 'roughness'],
    'normal': ['normal'],
    'opacity': ['opacity', 'alpha'],
    'height': ['height', 'displacement', 'bump'],
}

def find_texture_folders(base_path, suffix_filter=None):
    """Find all texture_* folders in the base path, optionally filtered by suffix"""
    folders = []
    for item in Path(base_path).iterdir():
        if item.is_dir() and item.name.startswith('texture_'):
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
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (400, 300)
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (700, 300)
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    y_offset = 600
    
    # Albedo/Base Color
    if textures.get('albedo'):
        tex_albedo = nodes.new(type='ShaderNodeTexImage')
        tex_albedo.location = (-400, y_offset)
        tex_albedo.label = "Albedo"
        tex_albedo.image = bpy.data.images.load(textures['albedo'])
        links.new(tex_albedo.outputs['Color'], bsdf.inputs['Base Color'])
        print(f"  ✓ Loaded Albedo")
        y_offset -= 300
    
    # Roughness
    if textures.get('roughness'):
        tex_rough = nodes.new(type='ShaderNodeTexImage')
        tex_rough.location = (-400, y_offset)
        tex_rough.label = "Roughness"
        tex_rough.image = bpy.data.images.load(textures['roughness'])
        tex_rough.image.colorspace_settings.name = 'Non-Color'
        links.new(tex_rough.outputs['Color'], bsdf.inputs['Roughness'])
        print(f"  ✓ Loaded Roughness")
        y_offset -= 300
    
    # Normal Map
    if textures.get('normal'):
        tex_normal = nodes.new(type='ShaderNodeTexImage')
        tex_normal.location = (-400, y_offset)
        tex_normal.label = "Normal"
        tex_normal.image = bpy.data.images.load(textures['normal'])
        tex_normal.image.colorspace_settings.name = 'Non-Color'
        
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (100, y_offset)
        
        links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
        print(f"  ✓ Loaded Normal")
        y_offset -= 300
    
    # Opacity/Alpha
    if textures.get('opacity'):
        tex_opacity = nodes.new(type='ShaderNodeTexImage')
        tex_opacity.location = (-400, y_offset)
        tex_opacity.label = "Opacity"
        tex_opacity.image = bpy.data.images.load(textures['opacity'])
        tex_opacity.image.colorspace_settings.name = 'Non-Color'
        links.new(tex_opacity.outputs['Color'], bsdf.inputs['Alpha'])
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'HASHED'
        print(f"  ✓ Loaded Opacity")
        y_offset -= 300
    
    # Height/Displacement
    if textures.get('height'):
        tex_height = nodes.new(type='ShaderNodeTexImage')
        tex_height.location = (-400, y_offset)
        tex_height.label = "Height"
        tex_height.image = bpy.data.images.load(textures['height'])
        tex_height.image.colorspace_settings.name = 'Non-Color'
        
        # Use as bump map
        bump = nodes.new(type='ShaderNodeBump')
        bump.location = (100, y_offset)
        bump.inputs['Strength'].default_value = 0.5
        
        links.new(tex_height.outputs['Color'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        print(f"  ✓ Loaded Height/Displacement")
    
    return mat

def create_plane_with_material(name, material, location):
    """Create a plane and apply material"""
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE, location=location)
    plane = bpy.context.active_object
    plane.name = name
    
    # Apply material
    if plane.data.materials:
        plane.data.materials[0] = material
    else:
        plane.data.materials.append(material)
    
    return plane

def process_all_textures():
    """Main function to process all texture folders"""
    base_path = Path(PBR_TEXTURES_FOLDER)
    
    if not base_path.exists():
        print(f"✗ Folder not found: {PBR_TEXTURES_FOLDER}")
        return
    
    # Find all texture folders with suffix filter
    texture_folders = find_texture_folders(base_path, FOLDER_SUFFIX)
    
    if not texture_folders:
        if FOLDER_SUFFIX:
            print(f"✗ No texture folders found with suffix '{FOLDER_SUFFIX}'!")
        else:
            print("✗ No texture folders found!")
        return
    
    print(f"Found {len(texture_folders)} texture sets")
    if FOLDER_SUFFIX:
        print(f"Filtered by suffix: {FOLDER_SUFFIX}")
    print("\n" + "="*70)
    
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    created_count = 0
    
    for idx, folder in enumerate(texture_folders):
        print(f"\n[{idx + 1}/{len(texture_folders)}] Processing: {folder.name}")
        
        # Find texture maps
        textures = {}
        for map_name, map_types in TEXTURE_MAP_TYPES.items():
            texture_path = find_texture_map(folder, map_types)
            if texture_path:
                textures[map_name] = texture_path
        
        if not textures:
            print(f"  ✗ No textures found in {folder.name}")
            continue
        
        # Calculate grid position
        row = idx // PLANES_PER_ROW
        col = idx % PLANES_PER_ROW
        location = (col * GRID_SPACING, -row * GRID_SPACING, 0)
        
        # Create material
        mat_name = f"Material_{folder.name}"
        material = create_pbr_material(mat_name, textures)
        
        # Create plane
        plane_name = f"Plane_{folder.name}"
        plane = create_plane_with_material(plane_name, material, location)
        
        print(f"  ✓ Created plane at ({location[0]:.1f}, {location[1]:.1f}, {location[2]:.1f})")
        created_count += 1
    
    # Set viewport shading to Material Preview
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
    
    # Fit view to show all objects
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = {'area': area, 'region': region}
                    bpy.ops.view3d.view_all(override)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total texture sets found: {len(texture_folders)}")
    print(f"Planes created: {created_count}")
    print(f"Grid layout: {PLANES_PER_ROW} planes per row")
    if FOLDER_SUFFIX:
        print(f"Suffix filter: {FOLDER_SUFFIX}")
    print("="*70)

# Run the script
if __name__ == "__main__":
    process_all_textures()