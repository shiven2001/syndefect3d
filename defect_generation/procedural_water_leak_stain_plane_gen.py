"""
Procedural water leak stain plane.
Water leak at the top half of the plane, with stains/drips flowing downward.
Based on defect_generation/procedural_spalling_plane_gen.py.
"""

import bpy
import random


def create_water_leak_stain(
    name="WaterLeakStainPlane",
):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_output = nodes.new("ShaderNodeOutputMaterial")
    node_bsdf_paint = nodes.new("ShaderNodeBsdfPrincipled")
    node_tex_coord = nodes.new("ShaderNodeTexCoord")

    # Mapping for scale/position
    node_mapping = nodes.new("ShaderNodeMapping")
    node_mapping.inputs["Scale"].default_value = (
        random.uniform(0.1, 0.5),
        random.uniform(1.0, 1.5),
        2.0,
    )
    node_mapping.inputs["Location"].default_value = (
        random.uniform(0.7, 0.9),
        random.uniform(1.0, 3.0),
        0.0,
    )

    #########################################################

    links.new(node_tex_coord.outputs["Object"], node_mapping.inputs["Vector"])

    node_gradient_1 = nodes.new("ShaderNodeTexGradient")
    node_gradient_1.gradient_type = "LINEAR"
    links.new(node_mapping.outputs["Vector"], node_gradient_1.inputs["Vector"])

    node_noise_1 = nodes.new("ShaderNodeTexNoise")
    node_noise_1.noise_dimensions = "2D"
    node_noise_1.normalize = True
    node_noise_1.noise_type = "FBM"
    node_noise_1.inputs["Scale"].default_value = random.uniform(5.0, 10.0)
    node_noise_1.inputs["Detail"].default_value = 2.0
    node_noise_1.inputs["Roughness"].default_value = 0.5
    links.new(node_mapping.outputs["Vector"], node_noise_1.inputs["Vector"])

    node_multiply_1 = nodes.new("ShaderNodeMath")
    node_multiply_1.operation = "MULTIPLY"
    node_multiply_1.inputs[1].default_value = random.uniform(0.4, 0.5)
    links.new(node_noise_1.outputs["Fac"], node_multiply_1.inputs[0])

    #########################################################

    node_invert_color_1 = nodes.new("ShaderNodeInvert")
    links.new(node_gradient_1.outputs["Color"], node_invert_color_1.inputs["Color"])

    node_add_1 = nodes.new("ShaderNodeMath")
    node_add_1.operation = "ADD"
    links.new(node_invert_color_1.outputs["Color"], node_add_1.inputs[0])
    links.new(node_multiply_1.outputs["Value"], node_add_1.inputs[1])

    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements.new(0.0)
    node_ramp_1.color_ramp.elements[0].position = 0.0
    node_ramp_1.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    node_ramp_1.color_ramp.elements[1].position = 0.5
    node_ramp_1.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    node_ramp_1.color_ramp.elements[2].position = 1.0
    node_ramp_1.color_ramp.elements[2].color = (
        0.0,
        0.0,
        0.0,
        1.0,
    )  # color of stain - brown to gray
    links.new(node_add_1.outputs["Value"], node_ramp_1.inputs["Fac"])

    node_ramp_2 = nodes.new("ShaderNodeValToRGB")
    node_ramp_2.color_ramp.elements.new(0.0)
    node_ramp_2.color_ramp.elements[0].position = random.uniform(0.5, 0.55)
    node_ramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node_ramp_2.color_ramp.elements[1].position = 0.9
    node_ramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    node_ramp_2.color_ramp.elements[2].position = 1.0
    node_ramp_2.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    links.new(node_add_1.outputs["Value"], node_ramp_2.inputs["Fac"])

    links.new(node_ramp_2.outputs["Color"], node_bsdf_paint.inputs["Alpha"])
    links.new(node_ramp_1.outputs["Color"], node_bsdf_paint.inputs["Base Color"])

    node_power_1 = nodes.new("ShaderNodeMath")
    node_power_1.operation = "POWER"
    node_power_1.inputs[1].default_value = 2.0
    links.new(node_ramp_2.outputs["Color"], node_power_1.inputs[0])

    node_multiply_1 = nodes.new("ShaderNodeMath")
    node_multiply_1.operation = "MULTIPLY"
    node_multiply_1.inputs[1].default_value = 100.0
    links.new(node_power_1.outputs["Value"], node_multiply_1.inputs[0])

    node_bump_1 = nodes.new("ShaderNodeBump")
    node_bump_1.inputs["Strength"].default_value = 0.2
    node_bump_1.inputs["Distance"].default_value = 2.0
    links.new(node_multiply_1.outputs["Value"], node_bump_1.inputs["Height"])
    links.new(node_bump_1.outputs["Normal"], node_bsdf_paint.inputs["Normal"])

    node_noise_2 = nodes.new("ShaderNodeTexNoise")
    node_noise_2.noise_dimensions = "3D"
    node_noise_2.normalize = True
    node_noise_2.noise_type = "FBM"
    node_noise_2.inputs["Scale"].default_value = random.uniform(5.0, 20.0)
    node_noise_2.inputs["Detail"].default_value = 5.0
    node_noise_2.inputs["Roughness"].default_value = 0.5

    node_ramp_3 = nodes.new("ShaderNodeValToRGB")
    node_ramp_3.color_ramp.elements[0].position = 0.0
    node_ramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node_ramp_3.color_ramp.elements[1].position = 1.0
    node_ramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(node_noise_2.outputs["Color"], node_ramp_3.inputs["Fac"])

    node_multiply_2 = nodes.new("ShaderNodeMath")
    node_multiply_2.operation = "MULTIPLY"
    node_multiply_2.inputs[1].default_value = 0.050
    links.new(node_ramp_3.outputs["Color"], node_multiply_2.inputs[0])

    node_bump_2 = nodes.new("ShaderNodeBump")
    node_bump_2.inputs["Strength"].default_value = 1.0
    node_bump_2.inputs["Distance"].default_value = 0.5
    links.new(node_multiply_2.outputs["Value"], node_bump_2.inputs["Height"])
    links.new(node_bump_2.outputs["Normal"], node_bump_1.inputs["Normal"])

    #########
    links.new(node_bsdf_paint.outputs["BSDF"], node_output.inputs["Surface"])

    mat.blend_method = "CLIP"
    mat.shadow_method = "NONE"

    return mat


def add_light():
    """Add a sun light to illuminate the plane from the front."""
    if "WaterLeakLight" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["WaterLeakLight"], do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 4))
    light = bpy.context.active_object
    light.name = "WaterLeakLight"
    light.data.energy = 2.0
    light.rotation_euler = (0, 0, 0)


def add_front_camera(distance=3.5):
    """Add a camera facing the front of the plane."""
    if "WaterLeakCamera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["WaterLeakCamera"], do_unlink=True)

    bpy.ops.object.camera_add(location=(0, 0, distance))
    camera = bpy.context.active_object
    camera.name = "WaterLeakCamera"
    cam_data = camera.data
    cam_data.sensor_fit = "VERTICAL"
    cam_data.sensor_width = 36
    cam_data.sensor_height = 36
    return camera


def create_water_leak_stain_plane():
    """Create a plane with water leak stain material."""
    if "WaterLeakStainPlane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["WaterLeakStainPlane"], do_unlink=True)

    bpy.ops.mesh.primitive_plane_add(size=2)
    plane = bpy.context.active_object
    plane.name = "WaterLeakStainPlane"
    bpy.ops.object.modifier_add(type="SUBSURF")
    subsurf = plane.modifiers["Subdivision"]
    subsurf.subdivision_type = "SIMPLE"
    subsurf.levels = 8
    subsurf.render_levels = 8

    material = create_water_leak_stain()
    plane.data.materials.append(material)

    add_light()
    add_front_camera(distance=3.5)

    if "WaterLeakCamera" in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects["WaterLeakCamera"]
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.context.scene.render.engine = "CYCLES"


if __name__ == "__main__":
    create_water_leak_stain_plane()
