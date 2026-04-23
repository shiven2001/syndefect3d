import bpy
import bmesh
import math
import random
from mathutils import noise, Vector


def get_fractal_noise(vec, octaves=3, scale=2.0):
    val = 0
    amp = 1.0
    freq = scale
    for _ in range(octaves):
        val += noise.noise(vec * freq) * amp
        amp *= 0.5
        freq *= 2.0
    return val


def get_distorted_displacement(x, y, bubbles, noise_scale=2.5, noise_strength=0.3):
    total_z = 0
    vec_x = Vector((x, y, 0.0))
    vec_y = Vector((y, x, 1.2))

    off_x = get_fractal_noise(vec_x, octaves=4, scale=noise_scale) * noise_strength
    off_y = get_fractal_noise(vec_y, octaves=4, scale=noise_scale) * noise_strength

    distorted_x = x + off_x
    distorted_y = y + off_y

    for b_x, b_y, radius, height in bubbles:
        dist = math.sqrt((distorted_x - b_x) ** 2 + (distorted_y - b_y) ** 2)
        if dist < radius:
            t = dist / radius
            displacement = height * math.cos((math.pi / 2) * t)
            total_z = max(total_z, displacement)

    return total_z


def create_distorted_blob_plane(
    size=2.0,
    subdivisions=2000,
    thickness=0.001,
    num_bubbles=6,
    edge_margin=0.25,
    min_radius=0.2,
    max_radius=0.5,
    min_height=0.05,
    max_height=0.08,
    distortion_scale=3.0,
    distortion_strength=0.4,
    seed=None,
):
    if seed is None:
        seed = random.randint(0, 1000)
    random.seed(seed)

    # --- 1. Material Setup ---
    def make_background_material(name):
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        node_output = nodes.new(type="ShaderNodeOutputMaterial")
        node_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

        node_bsdf.inputs["Alpha"].default_value = 0.0
        links.new(node_bsdf.outputs["BSDF"], node_output.inputs["Surface"])
        return mat

    def make_complex_material(name, scale=10.0, detail=2.0):
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()  # Clear default nodes to start fresh

        # 1. Create Nodes
        node_tex_coord = nodes.new(type="ShaderNodeTexCoord")
        node_noise = nodes.new(type="ShaderNodeTexNoise")
        node_ramp = nodes.new(type="ShaderNodeValToRGB")
        node_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        node_output = nodes.new(type="ShaderNodeOutputMaterial")

        # 2. Configure Noise Texture
        node_noise.inputs["Scale"].default_value = scale
        node_noise.inputs["Detail"].default_value = detail

        # 4. Link Nodes
        links.new(node_tex_coord.outputs["Object"], node_noise.inputs["Vector"])
        links.new(node_noise.outputs["Color"], node_ramp.inputs["Fac"])

        node_multiply = nodes.new(type="ShaderNodeMath")
        node_multiply.operation = "MULTIPLY"
        node_multiply.inputs[1].default_value = 0.05
        links.new(node_ramp.outputs["Color"], node_multiply.inputs[0])

        node_bump = nodes.new(type="ShaderNodeBump")
        node_bump.inputs["Strength"].default_value = 1.0
        node_bump.inputs["Distance"].default_value = 0.5
        links.new(node_multiply.outputs["Value"], node_bump.inputs["Height"])
        links.new(node_bump.outputs["Normal"], node_bsdf.inputs["Normal"])

        # links.new(node_ramp.outputs["Color"], node_bsdf.inputs["Base Color"])
        links.new(node_bsdf.outputs["BSDF"], node_output.inputs["Surface"])

        return mat

    # --- Define the materials with noise ---
    # Background Blue: High scale noise for fine "dusty" detail
    blue_mat = make_background_material(
        "blob_background",
    )

    # Bump White: Lower scale noise for broad "stain" variation
    white_mat = make_complex_material(
        "BumpWhite",
        scale=50.0,
    )

    # --- 2. Bubble Generation ---
    bubbles = []
    half_bound = (size / 2) - edge_margin
    attempts = 0
    while len(bubbles) < num_bubbles and attempts < 1000:
        attempts += 1
        r = random.uniform(min_radius, max_radius)
        limit = half_bound - r
        if limit <= 0:
            continue
        bx, by = random.uniform(-limit, limit), random.uniform(-limit, limit)
        if not any(
            math.sqrt((bx - ox) ** 2 + (by - oy) ** 2) < (r + orad) * 0.7
            for (ox, oy, orad, oh) in bubbles
        ):
            bh = random.uniform(min_height, max_height)
            bubbles.append((bx, by, r, bh))

    # --- 3. Geometry Construction ---
    bm = bmesh.new()
    n = subdivisions
    verts_front = []
    verts_back = []
    plane_half = size / 2

    for i in range(n + 1):
        row_f, row_b = [], []
        for j in range(n + 1):
            x = -plane_half + (j / n) * size
            y = -plane_half + (i / n) * size
            z_f = get_distorted_displacement(
                x, y, bubbles, distortion_scale, distortion_strength
            )

            v_f = bm.verts.new((x, y, z_f))
            v_b = bm.verts.new((x, y, -thickness))
            row_f.append(v_f)
            row_b.append(v_b)
        verts_front.append(row_f)
        verts_back.append(row_b)

    # Ensure we can set material indices
    bm.faces.ensure_lookup_table()

    # --- 4. Face Building & Coloring ---
    for i in range(n):
        for j in range(n):
            # Create Front Face
            f_face = bm.faces.new(
                (
                    verts_front[i][j],
                    verts_front[i][j + 1],
                    verts_front[i + 1][j + 1],
                    verts_front[i + 1][j],
                )
            )

            # Check if any vertex in this face has displacement
            # We use a tiny threshold (0.0001) to decide if it's "bumped"
            has_bump = any(v.co.z > 0.0001 for v in f_face.verts)
            f_face.material_index = 1 if has_bump else 0

            # Create Back Face (Always Blue)
            b_face = bm.faces.new(
                (
                    verts_back[i][j],
                    verts_back[i + 1][j],
                    verts_back[i + 1][j + 1],
                    verts_back[i][j + 1],
                )
            )
            b_face.material_index = 0

    # Side bridging (Blue)
    for i in range(n):
        sides = [
            (
                verts_front[i][0],
                verts_back[i][0],
                verts_back[i + 1][0],
                verts_front[i + 1][0],
            ),
            (
                verts_front[i][n],
                verts_front[i + 1][n],
                verts_back[i + 1][n],
                verts_back[i][n],
            ),
            (
                verts_front[0][i],
                verts_front[0][i + 1],
                verts_back[0][i + 1],
                verts_back[0][i],
            ),
            (
                verts_front[n][i],
                verts_back[n][i],
                verts_back[n][i + 1],
                verts_front[n][i + 1],
            ),
        ]
        for side in sides:
            s_face = bm.faces.new(side)
            s_face.material_index = 0

    mesh = bpy.data.meshes.new("ColoredDistortionMesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("ColoredDistortionPlane", mesh)
    bpy.context.collection.objects.link(obj)

    # Assign materials to slots
    obj.data.materials.append(blue_mat)  # Index 0
    obj.data.materials.append(white_mat)  # Index 1

    for poly in mesh.polygons:
        poly.use_smooth = True

    return obj


def add_overhead_area_light(
    name="WaterBubbleAreaLight",
    location=(0.0, 0.0, 3.0),
    size_x=3.0,
    size_y=3.0,
    power=1500.0,
    color=(1.0, 1.0, 1.0),
):
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    bpy.ops.object.light_add(type="AREA", location=location)
    light_obj = bpy.context.active_object
    light_obj.name = name

    light_data = light_obj.data
    light_data.shape = "RECTANGLE"
    light_data.size = size_x
    light_data.size_y = size_y
    light_data.energy = power
    light_data.color = color

    # Point straight down (-Z)
    light_obj.rotation_euler = (0.0, 0.0, 0.0)
    return light_obj


# Reset and Run
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

create_distorted_blob_plane(
    num_bubbles=2, distortion_scale=3.5, distortion_strength=0.5, subdivisions=1000
)

add_overhead_area_light(
    location=(0.0, 0.0, 3.0),
    size_x=3.0,
    size_y=3.0,
    power=200.0,
)
