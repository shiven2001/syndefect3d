from audioop import findmax
import bpy
import random

# need to findmax
# Randomization ranges:
# - mapping Scale x,y: 1 to 3
# - ramp_2 pos 0: 0.45 to 0.5, pos 1: pos_0 + 0.01 to 0.02
# - noise_1 scale: 3 to 5
# - noise_amount: 0.4 to 0.5
# - bump multiply: 50 to 150
# - wall bump: 0.02 to 0.08


def create_spalling(
    name="SpallingPlane",
):

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    #####

    node_tex = nodes.new("ShaderNodeTexCoord")
    node_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    node_out = nodes.new("ShaderNodeOutputMaterial")
    node_grad = nodes.new("ShaderNodeTexGradient")
    node_grad.gradient_type = "SPHERICAL"
    node_noise_1 = nodes.new("ShaderNodeTexNoise")
    node_noise_1.noise_dimensions = "3D"
    node_noise_1.normalize = True
    node_noise_1.noise_type = "FBM"
    node_noise_1.inputs["Scale"].default_value = random.uniform(3.0, 4.0)
    node_noise_1.inputs["Detail"].default_value = 10.0
    node_noise_1.inputs["Roughness"].default_value = 0.5

    #########################################################
    node_value = nodes.new("ShaderNodeValue")
    node_value.outputs["Value"].default_value = random.uniform(1.5, 4.0)
    node_multiply_vector = nodes.new("ShaderNodeVectorMath")
    node_multiply_vector.operation = "MULTIPLY"
    node_multiply_vector.inputs[1].default_value = (1, 1, 1)
    links.new(node_tex.outputs["Object"], node_multiply_vector.inputs[0])
    links.new(node_value.outputs["Value"], node_multiply_vector.inputs[1])

    node_mapping = nodes.new("ShaderNodeMapping")
    mapping_scale = random.uniform(1.8, 2.2)
    node_mapping.inputs["Scale"].default_value = (
        mapping_scale + random.uniform(-0.1, 0.1),
        mapping_scale + random.uniform(-0.1, 0.1),
        2.0,
    )
    links.new(node_multiply_vector.outputs["Vector"], node_mapping.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_noise_1.inputs["Vector"])
    links.new(node_mapping.outputs["Vector"], node_grad.inputs["Vector"])

    node_multiply_1 = nodes.new("ShaderNodeMath")
    node_multiply_1.operation = "MULTIPLY"
    node_multiply_1.inputs[1].default_value = 0.8
    node_multiply_1.use_clamp = True
    links.new(node_noise_1.outputs["Fac"], node_multiply_1.inputs[0])
    node_subtract_1 = nodes.new("ShaderNodeMath")
    node_subtract_1.operation = "SUBTRACT"
    links.new(node_noise_1.outputs["Fac"], node_subtract_1.inputs[1])
    links.new(node_grad.outputs["Color"], node_subtract_1.inputs[0])

    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements[0].position = random.uniform(
        0.00, 0.02
    )  # Tiny black core
    node_ramp_1.color_ramp.elements[0].color = (1, 1, 1, 1)
    node_ramp_1.color_ramp.elements[1].position = random.uniform(
        0.05, 0.10
    )  # Quick fade to floor
    node_ramp_1.color_ramp.elements[1].color = (0, 0, 0, 1)
    links.new(node_subtract_1.outputs["Value"], node_ramp_1.inputs["Fac"])
    links.new(node_ramp_1.outputs["Color"], node_bsdf.inputs["Base Color"])
    node_invert_1 = nodes.new("ShaderNodeInvert")
    node_invert_1.inputs["Fac"].default_value = 1.0
    links.new(node_ramp_1.outputs["Color"], node_invert_1.inputs["Color"])
    links.new(node_invert_1.outputs["Color"], node_bsdf.inputs["Alpha"])

    node_ramp_3 = nodes.new("ShaderNodeValToRGB")
    node_ramp_3.color_ramp.elements[0].position = 0.1
    v = random.uniform(0.4, 0.6)
    node_ramp_3.color_ramp.elements[0].color = (v, v, v, 1)
    node_ramp_3.color_ramp.elements[1].position = random.uniform(0.15, 0.2)
    node_ramp_3.color_ramp.elements[1].color = (0, 0, 0, 1)
    links.new(node_subtract_1.outputs["Value"], node_ramp_3.inputs["Fac"])

    ##### wall texture here

    node_noise_2 = nodes.new("ShaderNodeTexNoise")
    node_noise_2.noise_dimensions = "3D"
    node_noise_2.normalize = True
    node_noise_2.noise_type = "FBM"
    node_noise_2.inputs["Scale"].default_value = random.uniform(50.0, 100.0)
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
    node_multiply_2.inputs[1].default_value = 1.0

    links.new(node_ramp_2.outputs["Color"], node_multiply_2.inputs[0])

    node_bump_2 = nodes.new("ShaderNodeBump")
    node_bump_2.inputs["Strength"].default_value = 1.0
    node_bump_2.inputs["Distance"].default_value = 1.0
    links.new(node_multiply_2.outputs["Value"], node_bump_2.inputs["Height"])

    node_bump_1 = nodes.new("ShaderNodeBump")
    node_bump_1.inputs["Strength"].default_value = 1.0
    node_bump_1.inputs["Distance"].default_value = 1.0
    links.new(node_bump_2.outputs["Normal"], node_bump_1.inputs["Normal"])
    links.new(node_ramp_3.outputs["Color"], node_bump_1.inputs["Height"])
    links.new(node_bump_1.outputs["Normal"], node_bsdf.inputs["Normal"])
    links.new(node_bsdf.outputs["BSDF"], node_out.inputs["Surface"])
    # base here

    return mat


def add_light():
    """Add a sun light to illuminate the plane from the front."""
    if "SpallingLight" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["SpallingLight"], do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 4))
    light = bpy.context.active_object
    light.name = "SpallingLight"
    light.data.energy = 2.0
    # Point at the plane: rotate so light direction is -Z (towards plane at origin)
    light.rotation_euler = (0, 0, 0)


def add_front_camera(distance=3.5):
    """Add a camera facing the front of the plane. Square aspect ratio."""
    if "SpallingCamera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["SpallingCamera"], do_unlink=True)

    bpy.ops.object.camera_add(location=(0, 0, distance))
    camera = bpy.context.active_object
    camera.name = "SpallingCamera"

    # Square camera: equal sensor width and height
    cam_data = camera.data
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_width = 36
    cam_data.sensor_height = 36

    return camera


def create_spalling_plane():
    # Clean up
    if "SpallingPlane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["SpallingPlane"], do_unlink=True)

    # Create Plane with subdivision for displacement
    bpy.ops.mesh.primitive_plane_add(size=2)
    plane = bpy.context.active_object
    plane.name = "SpallingPlane"
    bpy.ops.object.modifier_add(type="SUBSURF")
    subsurf = plane.modifiers["Subdivision"]
    subsurf.subdivision_type = (
        "SIMPLE"  # Keep square shape; Catmull-Clark rounds corners
    )
    subsurf.levels = 10
    subsurf.render_levels = 10

    # Assign Material (with random parameters for variation)
    material = create_spalling()
    plane.data.materials.append(material)

    # Add light and camera
    add_light()
    add_front_camera(distance=3.5)

    # Set the new camera as active, square render resolution, and Cycles engine
    if "SpallingCamera" in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects["SpallingCamera"]
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.engine = "CYCLES"


create_spalling_plane()
