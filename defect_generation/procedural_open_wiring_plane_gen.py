import bpy
import random
import mathutils


def create_plane_with_hole_material(size=6, location=(0, 0, 0)):
    """Create a plane with a procedural hole material (spherical gradient)."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    plane = bpy.context.active_object

    hole_mat = bpy.data.materials.new(name="Hole_Material")
    hole_mat.use_nodes = True
    nodes = hole_mat.node_tree.nodes
    links = hole_mat.node_tree.links
    nodes.clear()

    node_tex = nodes.new("ShaderNodeTexCoord")
    node_bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    node_out = nodes.new("ShaderNodeOutputMaterial")
    node_grad = nodes.new("ShaderNodeTexGradient")
    node_grad.gradient_type = "SPHERICAL"
    node_noise_1 = nodes.new("ShaderNodeTexNoise")
    node_noise_1.noise_dimensions = "3D"
    node_noise_1.normalize = True
    node_noise_1.noise_type = "FBM"
    node_noise_1.inputs["Scale"].default_value = random.uniform(5.0, 10.0)
    node_noise_1.inputs["Detail"].default_value = 10.0
    node_noise_1.inputs["Roughness"].default_value = 0.5
    links.new(node_tex.outputs["Object"], node_noise_1.inputs["Vector"])

    #########################################################
    node_value = nodes.new("ShaderNodeValue")
    node_value.outputs["Value"].default_value = random.uniform(1.5, 4.0)
    node_multiply_vector = nodes.new("ShaderNodeVectorMath")
    node_multiply_vector.operation = "MULTIPLY"
    node_multiply_vector.inputs[1].default_value = (1, 1, 1)
    links.new(node_tex.outputs["Object"], node_multiply_vector.inputs[0])
    links.new(node_value.outputs["Value"], node_multiply_vector.inputs[1])
    links.new(node_multiply_vector.outputs["Vector"], node_grad.inputs["Vector"])

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
    plane.data.materials.append(hole_mat)

    return plane


def create_wire_scene():
    # 1. Clean the scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # 2. Create the Plane with a SMALLER Hole
    create_plane_with_hole_material(size=6, location=(0, 0, 0))

    # 3. Darker Wire Colors
    colors = [
        (0.2, 0.01, 0.01, 1),  # Deep Red
        (0.01, 0.01, 0.2, 1),  # Deep Blue
        (0.01, 0.15, 0.03, 1),  # Deep Green
        (0.25, 0.05, 0, 1),  # Deep Orange
        (0.2, 0.15, 0, 1),  # Deep Mustard/Yellow
    ]

    num_wires = random.randint(2, 4)

    for i in range(num_wires):
        # Material Setup (Realistic & Dark)
        rgba = random.choice(colors)
        wire_mat = bpy.data.materials.new(name=f"Wire_Mat_{i}")
        wire_mat.use_nodes = True
        w_nodes = wire_mat.node_tree.nodes
        w_links = wire_mat.node_tree.links

        bsdf = w_nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.4

        noise = w_nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 80
        bump = w_nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.15

        w_links.new(noise.outputs["Fac"], bump.inputs["Height"])
        w_links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Create the Wire Path
        curve_data = bpy.data.curves.new(name=f"Wire_Curve_{i}", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.fill_mode = "FULL"
        curve_data.bevel_depth = random.uniform(0.025, 0.045)
        curve_data.bevel_resolution = 6

        wire_obj = bpy.data.objects.new(f"Wire_Obj_{i}", curve_data)
        bpy.context.collection.objects.link(wire_obj)
        wire_obj.data.materials.append(wire_mat)

        spline = curve_data.splines.new("BEZIER")

        # REDUCED LENGTH: Fewer points and smaller offsets
        num_points = 5
        spline.bezier_points.add(num_points - 1)

        # Start exactly at center
        last_pos = mathutils.Vector((0, 0, 0))

        for p_idx in range(num_points):
            p = spline.bezier_points[p_idx]

            if p_idx == 0:
                p.co = last_pos
            else:
                # Smaller random offsets = shorter wires
                offset = mathutils.Vector(
                    (
                        random.uniform(-0.6, 0.6),
                        random.uniform(-0.6, 0.6),
                        random.uniform(0.4, 0.7),
                    )
                )
                p.co = last_pos + offset
                last_pos = p.co

            # Subtle handle randomization for organic curves
            h_size = 0.4
            p.handle_left = p.co + mathutils.Vector(
                (
                    random.uniform(-h_size, h_size),
                    random.uniform(-h_size, h_size),
                    random.uniform(-h_size, h_size),
                )
            )
            p.handle_right = p.co + mathutils.Vector(
                (
                    random.uniform(-h_size, h_size),
                    random.uniform(-h_size, h_size),
                    random.uniform(-h_size, h_size),
                )
            )


# Run script
create_wire_scene()
