import bpy
import random


# thickness = 0.001 to 0.004
# voroni scale = 0.45 to 2.0
# voroni randomness = 0.75 to 1.0
# noise_mix_fac = 0.5 to 0.8 (how much noise warps crack pattern)
# noise_scale = 1.0 to 2.5 (coordinate warping noise scale)
# mapping_offset = (x,y,z) random offset for crack placement
# bump_strength = 0.5 to 1.5 (crack depth)
# wall_bump_strength = 0.02 to 0.08 (wall surface roughness)
# base_color = (r,g,b,a) grey (0.26) to black (0)


def create_cracks(
    name="CrackPlane",
    thickness=0.003,
    voroni_scale=0.450,
    voroni_randomness=1.0,
    noise_mix_fac=0.65,
    noise_scale=1.5,
    mapping_offset=(0, 0, 0),
    bump_strength=1.0,
    wall_bump_strength=0.050,
    base_color=None,
):
    if base_color is None:
        v = random.uniform(0, 0.26)  # grey (0.26) to black (0)
        base_color = (v, v, v, 1.0)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    #####
    node_output = nodes.new("ShaderNodeOutputMaterial")

    node_tex_coord = nodes.new("ShaderNodeTexCoord")
    node_noise_1 = nodes.new("ShaderNodeTexNoise")

    node_noise_1.noise_dimensions = "3D"
    node_noise_1.normalize = True
    node_noise_1.noise_type = "FBM"
    node_noise_1.inputs["Scale"].default_value = noise_scale
    node_noise_1.inputs["Detail"].default_value = 5.0
    node_noise_1.inputs["Roughness"].default_value = 0.5

    links.new(node_tex_coord.outputs["Object"], node_noise_1.inputs["Vector"])

    node_color_mix_1 = nodes.new("ShaderNodeMixRGB")
    node_color_mix_1.blend_type = "MIX"
    node_color_mix_1.use_clamp = True

    node_color_mix_1.inputs["Fac"].default_value = noise_mix_fac

    links.new(node_noise_1.outputs["Color"], node_color_mix_1.inputs["Color1"])

    links.new(node_tex_coord.outputs["Object"], node_color_mix_1.inputs["Color2"])

    node_mapping_1 = nodes.new("ShaderNodeMapping")
    node_mapping_1.inputs["Location"].default_value[0] = mapping_offset[0]
    node_mapping_1.inputs["Location"].default_value[1] = mapping_offset[1]
    node_mapping_1.inputs["Location"].default_value[2] = mapping_offset[2]

    links.new(node_color_mix_1.outputs["Color"], node_mapping_1.inputs["Vector"])

    node_voronoi_1 = nodes.new("ShaderNodeTexVoronoi")
    node_voronoi_1.voronoi_dimensions = "3D"
    node_voronoi_1.feature = "DISTANCE_TO_EDGE"
    node_voronoi_1.inputs["Scale"].default_value = voroni_scale
    node_voronoi_1.inputs["Randomness"].default_value = voroni_randomness
    node_voronoi_1.inputs["Roughness"].default_value = 1.0
    node_voronoi_1.inputs["Detail"].default_value = 0.001

    links.new(node_mapping_1.outputs["Vector"], node_voronoi_1.inputs["Vector"])

    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements[0].position = 0.0
    node_ramp_1.color_ramp.elements[1].position = (
        thickness  # change for crack thickness
    )

    links.new(node_voronoi_1.outputs["Distance"], node_ramp_1.inputs["Fac"])

    node_bump_1 = nodes.new("ShaderNodeBump")
    node_bump_1.inputs["Strength"].default_value = bump_strength
    node_bump_1.inputs["Distance"].default_value = 1.0

    links.new(node_ramp_1.outputs["Color"], node_bump_1.inputs["Height"])

    node_bsdf_paint = nodes.new("ShaderNodeBsdfPrincipled")

    node_bsdf_paint.inputs["Base Color"].default_value = base_color

    links.new(node_bump_1.outputs["Normal"], node_bsdf_paint.inputs["Normal"])

    links.new(node_bsdf_paint.outputs["BSDF"], node_output.inputs["Surface"])

    noise_texture_2 = nodes.new("ShaderNodeTexNoise")
    noise_texture_2.noise_dimensions = "3D"
    noise_texture_2.normalize = True
    noise_texture_2.noise_type = "FBM"
    noise_texture_2.inputs["Scale"].default_value = 100.0
    noise_texture_2.inputs["Detail"].default_value = 2.0
    noise_texture_2.inputs["Roughness"].default_value = 0.5

    node_ramp_2 = nodes.new("ShaderNodeValToRGB")

    links.new(noise_texture_2.outputs["Color"], node_ramp_2.inputs["Fac"])

    node_multiply_1 = nodes.new("ShaderNodeMath")
    node_multiply_1.operation = "MULTIPLY"
    node_multiply_1.inputs[1].default_value = wall_bump_strength

    links.new(node_ramp_2.outputs["Color"], node_multiply_1.inputs[0])

    node_bump_2 = nodes.new("ShaderNodeBump")
    node_bump_2.inputs["Strength"].default_value = 1.0
    node_bump_2.inputs["Distance"].default_value = 1.0

    links.new(node_multiply_1.outputs["Value"], node_bump_2.inputs["Height"])

    links.new(node_bump_2.outputs["Normal"], node_bump_1.inputs["Normal"])

    # alpha
    node_invert_1 = nodes.new("ShaderNodeInvert")
    links.new(node_ramp_1.outputs["Color"], node_invert_1.inputs["Color"])
    links.new(node_invert_1.outputs["Color"], node_bsdf_paint.inputs["Alpha"])

    return mat


def add_light():
    """Add a sun light to illuminate the plane from the front."""
    if "CrackLight" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["CrackLight"], do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 4))
    light = bpy.context.active_object
    light.name = "CrackLight"
    light.data.energy = 2.0
    # Point at the plane: rotate so light direction is -Z (towards plane at origin)
    light.rotation_euler = (0, 0, 0)


def add_front_camera(distance=3.5):
    """Add a camera facing the front of the plane. Square aspect ratio."""
    if "CrackCamera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["CrackCamera"], do_unlink=True)

    bpy.ops.object.camera_add(location=(0, 0, distance))
    camera = bpy.context.active_object
    camera.name = "CrackCamera"

    # Square camera: equal sensor width and height
    cam_data = camera.data
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_width = 36
    cam_data.sensor_height = 36

    return camera


def create_cracked_plane():
    # Clean up
    if "CrackedPlane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["CrackedPlane"], do_unlink=True)

    # Create Plane with subdivision for displacement
    bpy.ops.mesh.primitive_plane_add(size=2)
    plane = bpy.context.active_object
    plane.name = "CrackedPlane"
    bpy.ops.object.modifier_add(type="SUBSURF")
    subsurf = plane.modifiers["Subdivision"]
    subsurf.subdivision_type = (
        "SIMPLE"  # Keep square shape; Catmull-Clark rounds corners
    )
    subsurf.levels = 10
    subsurf.render_levels = 10

    # Assign Material (with random parameters for variation)
    v = random.uniform(0, 0.26)  # grey to black
    material = create_cracks(
        base_color=(v, v, v, 1.0),
        thickness=random.uniform(0.001, 0.004),
        voroni_scale=random.uniform(0.45, 2.0),
        voroni_randomness=random.uniform(0.75, 1.0),
        noise_mix_fac=random.uniform(0.5, 0.8),
        noise_scale=random.uniform(1.0, 2.5),
        mapping_offset=(
            random.uniform(0, 100),
            random.uniform(0, 100),
            random.uniform(0, 100),
        ),
        bump_strength=random.uniform(0.5, 1.5),
        wall_bump_strength=random.uniform(0.02, 0.08),
    )
    plane.data.materials.append(material)

    # Add light and camera
    add_light()
    add_front_camera(distance=3.5)

    # Set the new camera as active, square render resolution, and Cycles engine
    if "CrackCamera" in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects["CrackCamera"]
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.engine = "CYCLES"


create_cracked_plane()
