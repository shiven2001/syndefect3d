import bpy
import random


# Randomized parameters (ranges):
# - noise_1 W: 1.0 to 2.0
# - ramp_1 pos 0: 0.4 to 0.45
# - ramp_1 pos 1: pos_0 + 0.01 to 0.05
# - bsdf base color: white to light grey
# - bump_1 Strength: 0.5 to 1.5
# - noise_2 Scale: 80 to 120
# - multiply_2 (wall bump): 0.03 to 0.08


def create_paint_peel(
    name="PaintPeelPlane",
):
    # Sample random parameters
    noise_1_w = random.uniform(1.0, 2.0)
    ramp_1_pos_0 = random.uniform(0.4, 0.45)
    ramp_1_pos_1 = ramp_1_pos_0 + random.uniform(0.01, 0.05)
    bump_1_strength = random.uniform(0.5, 1.5)
    noise_2_scale = random.uniform(80.0, 120.0)
    multiply_2_val = random.uniform(0.03, 0.08)
    base_color_val = random.uniform(0.82, 1.0)  # white to light grey

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    #####
    node_output = nodes.new("ShaderNodeOutputMaterial")

    node_tex_coord = nodes.new("ShaderNodeTexCoord")
    node_noise_1 = nodes.new("ShaderNodeTexNoise")

    node_noise_1.noise_dimensions = "4D"
    node_noise_1.normalize = False
    node_noise_1.noise_type = "FBM"
    ##
    node_noise_1.inputs["W"].default_value = noise_1_w
    node_noise_1.inputs["Scale"].default_value = 1.5
    node_noise_1.inputs["Detail"].default_value = 20.0
    node_noise_1.inputs["Roughness"].default_value = 0.5

    links.new(node_tex_coord.outputs["Object"], node_noise_1.inputs["Vector"])

    node_power_1 = nodes.new("ShaderNodeMath")
    node_power_1.operation = "POWER"
    node_power_1.inputs[1].default_value = 0.8

    links.new(node_noise_1.outputs["Color"], node_power_1.inputs[0])

    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements.new(
        1.000
    )  # Color Ramp has 2 elements by default; add 3rd
    node_ramp_1.color_ramp.elements[0].position = ramp_1_pos_0
    node_ramp_1.color_ramp.elements[1].position = min(ramp_1_pos_1, 0.99)
    node_ramp_1.color_ramp.elements[2].position = 1.000
    node_ramp_1.color_ramp.elements[0].color = (0, 0, 0, 1)
    node_ramp_1.color_ramp.elements[1].color = (1, 1, 1, 1)
    node_ramp_1.color_ramp.elements[2].color = (1, 1, 1, 1)

    links.new(node_power_1.outputs["Value"], node_ramp_1.inputs["Fac"])

    node_power_2 = nodes.new("ShaderNodeMath")
    node_power_2.operation = "POWER"
    node_power_2.inputs[1].default_value = 5.0

    links.new(node_ramp_1.outputs["Color"], node_power_2.inputs[0])

    node_multiply_1 = nodes.new("ShaderNodeMath")
    node_multiply_1.operation = "MULTIPLY"
    node_multiply_1.inputs[1].default_value = 2.0

    links.new(node_power_2.outputs["Value"], node_multiply_1.inputs[0])

    node_bump_1 = nodes.new("ShaderNodeBump")
    node_bump_1.invert = True
    node_bump_1.inputs["Strength"].default_value = bump_1_strength
    node_bump_1.inputs["Distance"].default_value = 20.0

    links.new(node_multiply_1.outputs["Value"], node_bump_1.inputs["Height"])

    ##### wall texture here

    node_noise_2 = nodes.new("ShaderNodeTexNoise")
    node_noise_2.noise_dimensions = "3D"
    node_noise_2.normalize = True
    node_noise_2.noise_type = "FBM"
    node_noise_2.inputs["Scale"].default_value = noise_2_scale
    node_noise_2.inputs["Detail"].default_value = 2.0
    node_noise_2.inputs["Roughness"].default_value = 0.5

    node_ramp_2 = nodes.new("ShaderNodeValToRGB")
    node_ramp_2.color_ramp.elements[0].position = 0.0
    node_ramp_2.color_ramp.elements[1].position = 1.0
    node_ramp_2.color_ramp.elements[0].color = (0, 0, 0, 1)
    node_ramp_2.color_ramp.elements[1].color = (1, 1, 1, 1)

    links.new(node_noise_2.outputs["Color"], node_ramp_2.inputs["Fac"])

    node_multiply_2 = nodes.new("ShaderNodeMath")
    node_multiply_2.operation = "MULTIPLY"
    node_multiply_2.inputs[1].default_value = multiply_2_val

    links.new(node_ramp_2.outputs["Color"], node_multiply_2.inputs[0])

    node_bump_2 = nodes.new("ShaderNodeBump")
    node_bump_2.inputs["Strength"].default_value = 1.0
    node_bump_2.inputs["Distance"].default_value = 1.0

    links.new(node_multiply_2.outputs["Value"], node_bump_2.inputs["Height"])

    links.new(node_bump_2.outputs["Normal"], node_bump_1.inputs["Normal"])

    node_bsdf_paint = nodes.new("ShaderNodeBsdfPrincipled")

    # Base color: random grey from white to black
    node_bsdf_paint.inputs["Base Color"].default_value = (
        base_color_val,
        base_color_val,
        base_color_val,
        1.0,
    )

    links.new(node_bump_1.outputs["Normal"], node_bsdf_paint.inputs["Normal"])

    links.new(node_bsdf_paint.outputs["BSDF"], node_output.inputs["Surface"])

    ######
    # alpha here
    links.new(node_ramp_1.outputs["Color"], node_bsdf_paint.inputs["Alpha"])

    return mat


def add_light():
    """Add a sun light to illuminate the plane from the front."""
    if "PaintPeelLight" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["PaintPeelLight"], do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 4))
    light = bpy.context.active_object
    light.name = "PaintPeelLight"
    light.data.energy = 2.0
    # Point at the plane: rotate so light direction is -Z (towards plane at origin)
    light.rotation_euler = (0, 0, 0)


def add_front_camera(distance=3.5):
    """Add a camera facing the front of the plane. Square aspect ratio."""
    if "PaintPeelCamera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["CrackCamera"], do_unlink=True)

    bpy.ops.object.camera_add(location=(0, 0, distance))
    camera = bpy.context.active_object
    camera.name = "PaintPeelCamera"

    # Square camera: equal sensor width and height
    cam_data = camera.data
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_width = 36
    cam_data.sensor_height = 36

    return camera


def create_paint_peel_plane():
    # Clean up
    if "PaintPeelPlane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["PaintPeelPlane"], do_unlink=True)

    # Create Plane with subdivision for displacement
    bpy.ops.mesh.primitive_plane_add(size=2)
    plane = bpy.context.active_object
    plane.name = "PaintPeelPlane"
    bpy.ops.object.modifier_add(type="SUBSURF")
    subsurf = plane.modifiers["Subdivision"]
    subsurf.subdivision_type = (
        "SIMPLE"  # Keep square shape; Catmull-Clark rounds corners
    )
    subsurf.levels = 10
    subsurf.render_levels = 10

    # Assign Material
    material = create_paint_peel()
    plane.data.materials.append(material)

    # Add light and camera
    add_light()
    add_front_camera(distance=3.5)

    # Set the new camera as active, square render resolution, and Cycles engine
    if "PaintPeelCamera" in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects["PaintPeelCamera"]
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.engine = "CYCLES"


create_paint_peel_plane()
