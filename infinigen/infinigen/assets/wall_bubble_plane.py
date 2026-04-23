#!/usr/bin/env python3
# Procedural wall-bubble plane asset for Infinigen (wall-mounted defect).
# Mesh-based bubbles from defect_generation/water_bubble.py (blister / bubble pattern).

import logging
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector, noise

from infinigen.assets.composition import material_assignments
from infinigen.assets.utils.object import new_bbox, new_plane
from infinigen.assets.utils.uv import unwrap_normal
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_canonical_surfaces
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.random import weighted_sample

logger = logging.getLogger(__name__)


def _modify_wall_mat_for_bubble(mat: bpy.types.Material) -> bpy.types.Material:
    """Copy material and modify for bubble: Add node for BSDF in Mix Shader slot 2, noise scale=1."""
    mat = mat.copy()
    if not mat.use_nodes or not mat.node_tree:
        return mat
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    # 1. If Principled BSDF output -> Mix Shader input 2, add Math Add between Base Color and its source
    for node in list(nodes):
        if node.type != "BSDF_PRINCIPLED":
            continue
        bsdf_out = node.outputs.get("BSDF")
        if not bsdf_out:
            continue
        # Check if BSDF is connected to Mix Shader input 2 (second shader)
        bsdf_links = [l for l in links if l.from_socket == bsdf_out]
        if not any(
            l.to_node.type == "MIX_SHADER" and l.to_socket == l.to_node.inputs[2]
            for l in bsdf_links
        ):
            continue
        base_color_in = node.inputs.get("Base Color")
        if not base_color_in:
            continue
        base_color_links = [l for l in links if l.to_socket == base_color_in]
        if not base_color_links:
            continue
        from_link = base_color_links[0]
        from_socket = from_link.from_socket
        # Insert Math Add: source -> input 1, (0,0,0) -> input 2, output -> Base Color
        add_node = nodes.new(type="ShaderNodeVectorMath")
        add_node.operation = "ADD"
        add_node.inputs[1].default_value = (0.0, 0.0, 0.0)
        links.remove(from_link)
        links.new(from_socket, add_node.inputs[0])
        links.new(add_node.outputs["Vector"], base_color_in)

    # 2. All noise nodes: scale = 1
    for node in nodes:
        if node.type != "TEX_NOISE":
            continue
        scale_in = node.inputs.get("Scale")
        if scale_in is not None:
            scale_in.default_value = 1.0

    return mat


# --- Mesh-based bubble logic (from defect_generation/water_bubble.py) ---


def _get_fractal_noise(vec: Vector, octaves: int = 3, scale: float = 2.0) -> float:
    val = 0.0
    amp = 1.0
    freq = scale
    for _ in range(octaves):
        val += noise.noise(vec * freq) * amp
        amp *= 0.5
        freq *= 2.0
    return val


def _get_distorted_displacement(
    x: float,
    y: float,
    bubbles: list,
    noise_scale: float = 2.5,
    noise_strength: float = 0.3,
) -> float:
    """Compute Z displacement for bubble bulges (cosine falloff from center)."""
    vec_x = Vector((x, y, 0.0))
    vec_y = Vector((y, x, 1.2))
    off_x = _get_fractal_noise(vec_x, octaves=4, scale=noise_scale) * noise_strength
    off_y = _get_fractal_noise(vec_y, octaves=4, scale=noise_scale) * noise_strength
    distorted_x = x + off_x
    distorted_y = y + off_y

    total_z = 0.0
    for b_x, b_y, radius, height in bubbles:
        dist = math.sqrt((distorted_x - b_x) ** 2 + (distorted_y - b_y) ** 2)
        if dist < radius:
            t = dist / radius
            displacement = height * math.cos((math.pi / 2) * t)
            total_z = max(total_z, displacement)
    return total_z


def create_distorted_blob_plane(
    size: float = 1.0,
    subdivisions: int = 400,
    thickness: float = 0.001,
    num_bubbles: int = 4,
    edge_margin: float = 0.15,
    min_radius: float = 0.08,
    max_radius: float = 0.2,
    min_height: float = 0.01,
    max_height: float = 0.04,
    distortion_scale: float = 3.0,
    distortion_strength: float = 0.3,
    seed: int = 0,
) -> bpy.types.Object:
    """
    Same as defect_generation/water_bubble.py create_distorted_blob_plane.
    Creates mesh with procedural bubble bulges, materials, and returns obj.
    """
    rng = np.random.default_rng(seed)

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

    # Use infinigen wall material (Plaster/Tile) for bubble surface - matches room walls
    with FixedSeed(seed):
        wall_mat_gen = weighted_sample(material_assignments.wall)()
        wall_mat = wall_mat_gen(vertical=True)

    blue_mat = make_background_material("WallBubbleBackground")
    # white_mat = make_complex_material("WallBubbleBump", scale=50.0)

    # --- 2. Bubble Generation ---
    bubbles = []
    half_bound = (size / 2) - edge_margin
    attempts = 0
    while len(bubbles) < num_bubbles and attempts < 500:
        attempts += 1
        r = float(rng.uniform(min_radius, max_radius))
        limit = half_bound - r
        if limit <= 0:
            continue
        bx = float(rng.uniform(-limit, limit))
        by = float(rng.uniform(-limit, limit))
        if not any(
            math.sqrt((bx - ox) ** 2 + (by - oy) ** 2) < (r + orad) * 0.7
            for (ox, oy, orad, _) in bubbles
        ):
            bh = float(rng.uniform(min_height, max_height))
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
            z_f = _get_distorted_displacement(
                x, y, bubbles, distortion_scale, distortion_strength
            )
            v_f = bm.verts.new((x, y, z_f))
            v_b = bm.verts.new((x, y, -thickness))
            row_f.append(v_f)
            row_b.append(v_b)
        verts_front.append(row_f)
        verts_back.append(row_b)

    bm.faces.ensure_lookup_table()

    # --- 4. Face Building & Coloring (same as water_bubble) ---
    for i in range(n):
        for j in range(n):
            f_face = bm.faces.new(
                (
                    verts_front[i][j],
                    verts_front[i][j + 1],
                    verts_front[i + 1][j + 1],
                    verts_front[i + 1][j],
                )
            )
            has_bump = any(v.co.z > 0.0001 for v in f_face.verts)
            f_face.material_index = 1 if has_bump else 0

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

    mesh = bpy.data.meshes.new("WallBubbleMesh")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("WallBubblePlane", mesh)
    bpy.context.collection.objects.link(obj)

    # UV unwrap for wall material (Plaster/Tile use UVMap)
    unwrap_normal(obj)

    obj.data.materials.append(blue_mat)  # Index 0
    obj.data.materials.append(wall_mat)  # Index 1

    for poly in mesh.polygons:
        poly.use_smooth = True

    return obj


class WallBubblePlaneFactory(AssetFactory):
    """
    Procedural wall-mounted 'wall bubble' plane:
    - Same geometry/orientation/embedding as other defect planes
    - Uses create_distorted_blob_plane (same as water_bubble.py)
    """

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Same thin vertical bbox as crack/paint-peel/spalling
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        geom_seed = int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))

        # Per-instance size and bubble-param variation
        with FixedSeed(geom_seed):
            scale_z_val = np.random.uniform(0.6, 1.0) * self.plane_size / 2
            scale_y_val = np.random.uniform(0.6, 1.0) * self.plane_size / 2
            num_bubbles = int(np.random.randint(2, 7))
            min_radius = float(np.random.uniform(0.04, 0.08))
            max_radius = float(np.random.uniform(0.12, 0.2))
            max_radius = max(max_radius, min_radius + 0.02)
            min_height = float(np.random.uniform(0.004, 0.01))
            max_height = float(np.random.uniform(0.01, 0.02))
            max_height = max(max_height, min_height + 0.003)
            distortion_scale = float(np.random.uniform(2.0, 4.0))
            distortion_strength = float(np.random.uniform(0.2, 0.4))

        # Same as water_bubble.py: create_distorted_blob_plane returns obj
        plane = create_distorted_blob_plane(
            size=1.0,
            subdivisions=1000,
            thickness=0.001,
            num_bubbles=num_bubbles,
            edge_margin=0.15,
            min_radius=min_radius,
            max_radius=max_radius,
            min_height=min_height,
            max_height=max_height,
            distortion_scale=distortion_scale,
            distortion_strength=distortion_strength,
            seed=geom_seed,
        )
        plane.name = f"WallBubblePlane_{geom_seed}"

        plane.scale = (scale_z_val, scale_y_val, 1.0)
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)

        plane.visible_shadow = True
        plane.visible_diffuse = True
        return plane

    def finalize_assets(
        self, assets, state=None, wall_by_name=None, update_embed_transform=True
    ):
        """
        Position wall-bubble planes slightly ahead of the wall (toward the room),
        so the bubble bulge is more visible. Copy the wall's material for the
        bubble surface so colors match the wall.
        Uses state + wall_by_name to get the correct wall per bubble (avoids wrong room material).
        """
        EMBED_OFFSET = -0.03  # negative = outward from wall (more ahead)

        from infinigen.core import tags as t
        from infinigen.core.tagging import tagged_face_mask

        # Build wall_by_name if not provided
        if wall_by_name is None:
            wall_by_name = {
                w.name: w
                for w in bpy.data.objects
                if w.name.endswith(".wall") and w.type == "MESH"
            }

        def _get_wall_mat_for_bubble(bubble_obj):
            """Get material from the wall this bubble is placed against."""
            if state is not None and wall_by_name:
                for objkey, os in state.objs.items():
                    if os.obj is not bubble_obj:
                        continue
                    for rel in os.relations:
                        room_name = rel.target_name
                        if (
                            room_name in state.objs
                            and t.Semantics.Room in state.objs[room_name].tags
                        ):
                            wall_name = room_name.split(".")[0] + ".wall"
                            if wall_name in wall_by_name:
                                wall_obj = wall_by_name[wall_name]
                                for mat in wall_obj.data.materials:
                                    if mat is not None:
                                        return mat
                    break
            # Fallback: nearest wall material by world-space center.
            # This path is used during render-time resampling where `state` is unavailable.
            nearest_wall = None
            best_d2 = None
            bubble_center = bubble_obj.matrix_world.translation
            for w in wall_by_name.values():
                wall_center = w.matrix_world.translation
                d2 = (bubble_center - wall_center).length_squared
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    nearest_wall = w
            if nearest_wall is not None:
                for mat in nearest_wall.data.materials:
                    if mat is not None:
                        return mat

            # Last fallback: first wall material
            for w in wall_by_name.values():
                for mat in w.data.materials:
                    if mat is not None:
                        return mat
            return None

        # Fallback material when no wall found
        fallback_mat = None

        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                # Copy wall material to bubble surface (slot 1) - use correct wall for this bubble
                wall_mat = _get_wall_mat_for_bubble(obj)
                if wall_mat is None:
                    if fallback_mat is None:
                        fallback_mat = bpy.data.materials.get(
                            "WallBubbleFallbackRed"
                        ) or bpy.data.materials.new("WallBubbleFallbackRed")
                        fallback_mat.use_nodes = True
                        nodes = fallback_mat.node_tree.nodes
                        links = fallback_mat.node_tree.links
                        nodes.clear()
                        node_output = nodes.new(type="ShaderNodeOutputMaterial")
                        node_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
                        node_bsdf.inputs["Base Color"].default_value = (
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                        )
                        links.new(
                            node_bsdf.outputs["BSDF"], node_output.inputs["Surface"]
                        )
                    wall_mat = fallback_mat
                if wall_mat is not None and len(obj.data.materials) > 1:
                    seed = int_hash(obj.name)
                    bubble_mat = _modify_wall_mat_for_bubble(wall_mat)
                    bubble_mat.name = f"BubbleMaterial_{seed}"
                    obj.data.materials[1] = bubble_mat

                if update_embed_transform:
                    back_mask = tagged_face_mask(obj, {t.Subpart.Back})
                    if back_mask.any():
                        back_faces = [i for i, tag in enumerate(back_mask) if tag]
                        if back_faces:
                            largest_back_face_idx = max(
                                back_faces, key=lambda idx: obj.data.polygons[idx].area
                            )
                            back_poly = obj.data.polygons[largest_back_face_idx]
                        else:
                            continue
                    else:
                        back_poly = max(
                            obj.data.polygons,
                            key=lambda p: -p.normal.y if p.normal.y < 0 else -1e6,
                        )
                    wall_normal = np.array(butil.global_polygon_normal(obj, back_poly))
                    wall_normal = Vector(wall_normal).normalized()

                    translation = Vector(wall_normal * EMBED_OFFSET)
                    obj.location += translation
            except Exception as e:
                logger.warning("Failed to embed wall-bubble plane %s: %s", obj.name, e)


def refresh_wall_bubble_materials(wall_objects):
    """
    Re-sync wall-bubble materials to current wall materials without moving geometry.
    Call this after room wall materials are re-sampled in render mode.
    """
    bubbles = [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and o.name.startswith("WallBubblePlane")
    ]
    if not bubbles:
        return

    wall_by_name = {w.name: w for w in wall_objects if w and w.type == "MESH"}
    if not wall_by_name:
        return

    # Use finalize logic with transform updates disabled to avoid cumulative drift.
    WallBubblePlaneFactory(factory_seed=0).finalize_assets(
        bubbles, state=None, wall_by_name=wall_by_name, update_embed_transform=False
    )
