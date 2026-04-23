# Copyright (C) 2023, Princeton University.
# Procedural spalling plane asset for Infinigen (wall-mounted, defect semantics).
# Adapted from defect_generation/procedural_spalling_plane_gen.py.
# SpallingPlugPlaneFactory uses mesh-based logic from defect_generation/wall_plug_hole.py.

# need to fix

import logging
import math

import bpy
import bmesh
import numpy as np
from mathutils import Vector, noise

from infinigen.assets.utils.object import new_bbox, new_plane
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_canonical_surfaces, tagged_face_mask
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core import tags as t

logger = logging.getLogger(__name__)


# --- Wall plug hole mesh logic (from defect_generation/wall_plug_hole.py) ---


def _get_spalling_depth(x, y, chips, noise_scale=8.0, noise_strength=0.15):
    """Calculates a jagged, steep-walled indentation with a rocky floor."""
    max_d = 0.0
    nv = Vector((x * noise_scale, y * noise_scale, 0))
    dx = x + noise.noise(nv) * noise_strength
    dy = y + noise.noise(nv + Vector((10, 10, 0))) * noise_strength
    floor_noise = noise.noise(Vector((x * 20, y * 20, 0))) * 0.003

    for cx, cy, radius, depth in chips:
        dist = math.sqrt((dx - cx) ** 2 + (dy - cy) ** 2)
        if dist < radius:
            t = dist / radius
            falloff = 1.0 - (t**6)
            indent = (depth * falloff) + (floor_noise * falloff)
            max_d = max(max_d, indent)

    return max_d


def create_spalling_plug_mesh(
    seed: int,
    size: float = 1.0,
    subdivisions: int = 120,
    num_chips: int = 4,
    center_bounds_frac: float = 0.2,
    chip_radius_min: float = 0.1,
    chip_radius_max: float = 0.2,
    chip_depth: float = 0.05,
    noise_scale: float = 8.0,
    noise_strength: float = 0.15,
) -> bpy.types.Object:
    """
    Create a single-layer mesh with jagged spalling indentations (holes).
    Based on defect_generation/wall_plug_hole.py create_centered_spalling_mesh().
    """
    with FixedSeed(seed):
        chips = []
        center_bounds = size * center_bounds_frac
        for _ in range(num_chips):
            cx = float(np.random.uniform(-center_bounds, center_bounds))
            cy = float(np.random.uniform(-center_bounds, center_bounds))
            r = float(np.random.uniform(chip_radius_min, chip_radius_max))
            chips.append((cx, cy, r, chip_depth))

    bm = bmesh.new()
    n = subdivisions
    half = size / 2
    grid = []

    for i in range(n + 1):
        row = []
        for j in range(n + 1):
            x = -half + (j / n) * size
            y = -half + (i / n) * size
            indent = _get_spalling_depth(x, y, chips, noise_scale, noise_strength)
            v = bm.verts.new((x, y, -indent))
            row.append(v)
        grid.append(row)

    for i in range(n):
        for j in range(n):
            bm.faces.new(
                (grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j])
            )

    mesh = bpy.data.meshes.new(f"SpallingPlugMesh_{seed}")
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(f"SpallingPlugPlane_{seed}", mesh)
    bpy.context.collection.objects.link(obj)

    for poly in mesh.polygons:
        poly.use_smooth = True

    return obj


def create_spalling_plug_mesh_material(
    name: str,
    seed: int,
) -> bpy.types.Material:
    """
    Spalling material for mesh-based spalling plug.
    Same concept as create_spalling_material but node_value_val 2.5-3.0 for plug hole size.
    """
    with FixedSeed(seed):
        noise_1_scale = np.random.uniform(3.0, 4.0)
        node_value_val = np.random.uniform(2.5, 3.0)  # can lower high bound
        mapping_scale = np.random.uniform(1.8, 2.2)
        mapping_scale_x = mapping_scale + np.random.uniform(-0.1, 0.1)
        mapping_scale_y = mapping_scale + np.random.uniform(-0.1, 0.1)
        ramp_1_pos_0 = np.random.uniform(0.00, 0.02)
        ramp_1_pos_1 = np.random.uniform(0.05, 0.10)
        ramp_3_pos_0 = 0.1
        ramp_3_color_v = np.random.uniform(0.4, 0.6)
        ramp_3_pos_1 = np.random.uniform(0.15, 0.2)
        noise_2_scale = np.random.uniform(50.0, 100.0)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
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
    node_noise_1.inputs["Scale"].default_value = noise_1_scale
    node_noise_1.inputs["Detail"].default_value = 10.0
    node_noise_1.inputs["Roughness"].default_value = 0.5

    node_value = nodes.new("ShaderNodeValue")
    node_value.outputs["Value"].default_value = node_value_val
    node_combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(node_value.outputs["Value"], node_combine.inputs["X"])
    links.new(node_value.outputs["Value"], node_combine.inputs["Y"])
    links.new(node_value.outputs["Value"], node_combine.inputs["Z"])
    node_multiply_vector = nodes.new("ShaderNodeVectorMath")
    node_multiply_vector.operation = "MULTIPLY"
    links.new(node_tex.outputs["Object"], node_multiply_vector.inputs[0])
    links.new(node_combine.outputs["Vector"], node_multiply_vector.inputs[1])

    node_mapping = nodes.new("ShaderNodeMapping")
    node_mapping.inputs["Scale"].default_value = (mapping_scale_x, mapping_scale_y, 2.0)
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
    node_ramp_1.color_ramp.elements[0].position = ramp_1_pos_0
    node_ramp_1.color_ramp.elements[0].color = (1, 1, 1, 1)
    node_ramp_1.color_ramp.elements[1].position = ramp_1_pos_1
    node_ramp_1.color_ramp.elements[1].color = (0, 0, 0, 1)
    links.new(node_subtract_1.outputs["Value"], node_ramp_1.inputs["Fac"])
    links.new(node_ramp_1.outputs["Color"], node_bsdf.inputs["Base Color"])
    node_invert_1 = nodes.new("ShaderNodeInvert")
    node_invert_1.inputs["Fac"].default_value = 1.0
    links.new(node_ramp_1.outputs["Color"], node_invert_1.inputs["Color"])
    links.new(node_invert_1.outputs["Color"], node_bsdf.inputs["Alpha"])

    node_ramp_3 = nodes.new("ShaderNodeValToRGB")
    node_ramp_3.color_ramp.elements[0].position = ramp_3_pos_0
    node_ramp_3.color_ramp.elements[0].color = (
        ramp_3_color_v,
        ramp_3_color_v,
        ramp_3_color_v,
        1,
    )
    node_ramp_3.color_ramp.elements[1].position = ramp_3_pos_1
    node_ramp_3.color_ramp.elements[1].color = (0, 0, 0, 1)
    links.new(node_subtract_1.outputs["Value"], node_ramp_3.inputs["Fac"])

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

    mat.blend_method = "CLIP"
    mat.shadow_method = "NONE"
    return mat


# --- Shader-based spalling (generic SpallingPlaneFactory) ---
def create_spalling_material(
    name: str,
    seed: int,
) -> bpy.types.Material:
    """Transparent plane with a single, central, irregular 'spalling' hole.
    Adapted from defect_generation/procedural_spalling_plane_gen.py create_spalling().
    Same randomizations: noise scale, node_value, mapping, ramp positions, wall texture.
    """
    with FixedSeed(seed):
        noise_1_scale = np.random.uniform(3.0, 4.0)
        node_value_val = np.random.uniform(2.5, 8.0)
        mapping_scale = np.random.uniform(1.8, 2.2)
        mapping_scale_x = mapping_scale + np.random.uniform(-0.1, 0.1)
        mapping_scale_y = mapping_scale + np.random.uniform(-0.1, 0.1)
        ramp_1_pos_0 = np.random.uniform(0.00, 0.02)
        ramp_1_pos_1 = np.random.uniform(0.05, 0.10)
        ramp_3_pos_0 = 0.1
        ramp_3_color_v = np.random.uniform(0.4, 0.6)
        ramp_3_pos_1 = np.random.uniform(0.15, 0.2)
        noise_2_scale = np.random.uniform(50.0, 100.0)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
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
    node_noise_1.inputs["Scale"].default_value = noise_1_scale
    node_noise_1.inputs["Detail"].default_value = 10.0
    node_noise_1.inputs["Roughness"].default_value = 0.5

    node_value = nodes.new("ShaderNodeValue")
    node_value.outputs["Value"].default_value = node_value_val
    node_combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(node_value.outputs["Value"], node_combine.inputs["X"])
    links.new(node_value.outputs["Value"], node_combine.inputs["Y"])
    links.new(node_value.outputs["Value"], node_combine.inputs["Z"])
    node_multiply_vector = nodes.new("ShaderNodeVectorMath")
    node_multiply_vector.operation = "MULTIPLY"
    links.new(node_tex.outputs["Object"], node_multiply_vector.inputs[0])
    links.new(node_combine.outputs["Vector"], node_multiply_vector.inputs[1])

    node_mapping = nodes.new("ShaderNodeMapping")
    node_mapping.inputs["Scale"].default_value = (mapping_scale_x, mapping_scale_y, 2.0)
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
    node_ramp_1.color_ramp.elements[0].position = ramp_1_pos_0
    node_ramp_1.color_ramp.elements[0].color = (1, 1, 1, 1)
    node_ramp_1.color_ramp.elements[1].position = ramp_1_pos_1
    node_ramp_1.color_ramp.elements[1].color = (0, 0, 0, 1)
    links.new(node_subtract_1.outputs["Value"], node_ramp_1.inputs["Fac"])
    links.new(node_ramp_1.outputs["Color"], node_bsdf.inputs["Base Color"])
    node_invert_1 = nodes.new("ShaderNodeInvert")
    node_invert_1.inputs["Fac"].default_value = 1.0
    links.new(node_ramp_1.outputs["Color"], node_invert_1.inputs["Color"])
    links.new(node_invert_1.outputs["Color"], node_bsdf.inputs["Alpha"])

    node_ramp_3 = nodes.new("ShaderNodeValToRGB")
    node_ramp_3.color_ramp.elements[0].position = ramp_3_pos_0
    node_ramp_3.color_ramp.elements[0].color = (
        ramp_3_color_v,
        ramp_3_color_v,
        ramp_3_color_v,
        1,
    )
    node_ramp_3.color_ramp.elements[1].position = ramp_3_pos_1
    node_ramp_3.color_ramp.elements[1].color = (0, 0, 0, 1)
    links.new(node_subtract_1.outputs["Value"], node_ramp_3.inputs["Fac"])

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

    mat.blend_method = "CLIP"
    mat.shadow_method = "NONE"
    return mat


def create_spalling_plug_material_shader(
    name: str,
    seed: int,
) -> bpy.types.Material:
    """
    [Legacy] Spalling hole material variant for plug-level defects (shader-based).
    Kept for reference; SpallingPlugPlaneFactory now uses create_spalling_plug_mesh + create_spalling_plug_mesh_material.
    """

    mapping_location = (0.0, 0.0, 0.0)
    with FixedSeed(seed):
        mapping_scale_x = np.random.uniform(1.0, 3.0)
        mapping_scale_y = np.random.uniform(1.0, 3.0)
        noise_1_scale = np.random.uniform(3.0, 5.0)
        mapping_location = (0.0, 0.0, 0.0)
        noise_amount = np.random.uniform(-0.5, -1.0)
        ramp_2_pos_0 = np.random.uniform(0.45, 0.5)
        ramp_2_pos_1 = ramp_2_pos_0 + np.random.uniform(0.01, 0.02)
        power_exponent = np.random.uniform(1.5, 2.5)
        bump_multiply = np.random.uniform(50.0, 150.0)
        bump_strength = np.random.uniform(0.7, 1.3)
        bump_distance = np.random.uniform(15.0, 25.0)
        wall_noise_scale = np.random.uniform(80.0, 120.0)
        wall_bump_strength = np.random.uniform(0.02, 0.08)
        v = np.random.uniform(0, 0.08)
        base_color_inner = (v, v, v, 1.0)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_output = nodes.new("ShaderNodeOutputMaterial")
    node_bsdf_paint = nodes.new("ShaderNodeBsdfPrincipled")
    node_tex_coord = nodes.new("ShaderNodeTexCoord")

    node_mapping = nodes.new("ShaderNodeMapping")
    node_mapping.inputs["Scale"].default_value = (
        mapping_scale_x,
        mapping_scale_y,
        2.0,
    )
    node_mapping.inputs["Location"].default_value[0] = mapping_location[0]
    node_mapping.inputs["Location"].default_value[1] = mapping_location[1]
    node_mapping.inputs["Location"].default_value[2] = mapping_location[2]
    links.new(node_tex_coord.outputs["Object"], node_mapping.inputs["Vector"])

    node_noise_1 = nodes.new("ShaderNodeTexNoise")
    links.new(node_mapping.outputs["Vector"], node_noise_1.inputs["Vector"])
    node_gradient_1 = nodes.new("ShaderNodeTexGradient")
    links.new(node_mapping.outputs["Vector"], node_gradient_1.inputs["Vector"])

    node_noise_1.noise_dimensions = "2D"
    node_noise_1.normalize = True
    node_noise_1.noise_type = "FBM"
    node_noise_1.inputs["Scale"].default_value = noise_1_scale
    node_noise_1.inputs["Detail"].default_value = 10.0
    node_noise_1.inputs["Roughness"].default_value = 0.5

    node_gradient_1.gradient_type = "SPHERICAL"

    node_multiply_1 = nodes.new("ShaderNodeMath")
    node_multiply_1.operation = "MULTIPLY"
    node_multiply_1.inputs[1].default_value = noise_amount
    links.new(node_noise_1.outputs["Fac"], node_multiply_1.inputs[0])

    node_add_1 = nodes.new("ShaderNodeMath")
    node_add_1.operation = "ADD"
    links.new(node_gradient_1.outputs["Fac"], node_add_1.inputs[0])
    links.new(node_multiply_1.outputs["Value"], node_add_1.inputs[1])

    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements.new(0.0)
    node_ramp_1.color_ramp.elements[0].position = 0.0
    node_ramp_1.color_ramp.elements[0].color = (1, 1, 1, 1)
    node_ramp_1.color_ramp.elements[1].position = 0.5
    node_ramp_1.color_ramp.elements[1].color = base_color_inner
    node_ramp_1.color_ramp.elements[2].position = 1.0
    node_ramp_1.color_ramp.elements[2].color = base_color_inner
    links.new(node_add_1.outputs["Value"], node_ramp_1.inputs["Fac"])

    node_ramp_2 = nodes.new("ShaderNodeValToRGB")
    node_ramp_2.color_ramp.elements.new(1.0)
    node_ramp_2.color_ramp.elements[0].position = ramp_2_pos_0
    node_ramp_2.color_ramp.elements[1].position = min(ramp_2_pos_1, 0.99)
    node_ramp_2.color_ramp.elements[2].position = 1.0
    node_ramp_2.color_ramp.elements[0].color = (0, 0, 0, 1)
    node_ramp_2.color_ramp.elements[1].color = (1, 1, 1, 1)
    node_ramp_2.color_ramp.elements[2].color = (1, 1, 1, 1)

    links.new(node_add_1.outputs["Value"], node_ramp_2.inputs["Fac"])

    node_power_1 = nodes.new("ShaderNodeMath")
    node_power_1.operation = "POWER"
    node_power_1.use_clamp = False
    node_power_1.inputs[1].default_value = power_exponent
    links.new(node_ramp_2.outputs["Color"], node_power_1.inputs[0])

    links.new(node_ramp_2.outputs["Color"], node_bsdf_paint.inputs["Alpha"])

    node_multiply_bump = nodes.new("ShaderNodeMath")
    node_multiply_bump.operation = "MULTIPLY"
    node_multiply_bump.use_clamp = False
    node_multiply_bump.inputs[1].default_value = bump_multiply
    links.new(node_power_1.outputs["Value"], node_multiply_bump.inputs[0])

    node_bump_1 = nodes.new("ShaderNodeBump")
    node_bump_1.invert = False
    node_bump_1.inputs["Strength"].default_value = bump_strength
    node_bump_1.inputs["Distance"].default_value = bump_distance
    links.new(node_multiply_bump.outputs["Value"], node_bump_1.inputs["Height"])

    node_noise_2 = nodes.new("ShaderNodeTexNoise")
    node_noise_2.noise_dimensions = "3D"
    node_noise_2.normalize = True
    node_noise_2.noise_type = "FBM"
    node_noise_2.inputs["Scale"].default_value = wall_noise_scale
    node_noise_2.inputs["Detail"].default_value = 2.0
    node_noise_2.inputs["Roughness"].default_value = 0.5

    node_ramp_wall = nodes.new("ShaderNodeValToRGB")
    node_ramp_wall.color_ramp.elements[0].position = 0.0
    node_ramp_wall.color_ramp.elements[1].position = 1.0
    node_ramp_wall.color_ramp.elements[0].color = (0, 0, 0, 1)
    node_ramp_wall.color_ramp.elements[1].color = (1, 1, 1, 1)

    links.new(node_noise_2.outputs["Color"], node_ramp_wall.inputs["Fac"])

    node_multiply_2 = nodes.new("ShaderNodeMath")
    node_multiply_2.operation = "MULTIPLY"
    node_multiply_2.inputs[1].default_value = wall_bump_strength

    links.new(node_ramp_wall.outputs["Color"], node_multiply_2.inputs[0])

    node_bump_2 = nodes.new("ShaderNodeBump")
    node_bump_2.inputs["Strength"].default_value = 1.0
    node_bump_2.inputs["Distance"].default_value = 1.0

    links.new(node_multiply_2.outputs["Value"], node_bump_2.inputs["Height"])
    links.new(node_bump_2.outputs["Normal"], node_bump_1.inputs["Normal"])

    links.new(node_bump_1.outputs["Normal"], node_bsdf_paint.inputs["Normal"])
    links.new(node_bsdf_paint.outputs["BSDF"], node_output.inputs["Surface"])
    links.new(node_ramp_1.outputs["Color"], node_bsdf_paint.inputs["Base Color"])

    mat.blend_method = "CLIP"
    mat.shadow_method = "CLIP"

    return mat


class SpallingPlaneFactory(AssetFactory):
    """Procedural wall-mounted spalling plane with an irregular central hole."""

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Same vertical bbox geometry as crack/paint-peel planes
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        plane = new_plane()

        with FixedSeed(int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))):
            scale_z_val = np.random.uniform(0.6, 1.0) * self.plane_size / 2
            scale_y_val = np.random.uniform(0.6, 1.0) * self.plane_size / 2

        plane.scale = (scale_z_val, scale_y_val, 1)
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)

        mat = create_spalling_material(
            name=f"SpallingMaterial_{id(plane)}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
        )
        plane.data.materials.append(mat)
        plane.visible_shadow = False
        plane.visible_diffuse = False
        return plane


class SpallingPlugPlaneFactory(AssetFactory):
    """Spalling plane variant used specifically at wall plug locations."""

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Same vertical bbox geometry; still need canonical tags for wall/flush constraints
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        seed_geom = int_hash((self.factory_seed, kwargs.get("i", 0), "geom_plug"))
        seed_mat = int_hash((self.factory_seed, kwargs.get("i", 0), "plug"))

        with FixedSeed(seed_geom):
            # Same scale logic as before (different from regular spalling - smaller for plug)
            scale_z_val = 0.7 * np.random.uniform(0.6, 1.0) * self.plane_size / 2
            scale_y_val = 0.7 * np.random.uniform(0.6, 1.0) * self.plane_size / 2

        # Same planar mesh as SpallingPlaneFactory (flat plane, no deformation)
        plane = new_plane()

        plane.scale = (scale_z_val, scale_y_val, 1)
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)

        mat = create_spalling_material(
            name=f"SpallingPlugMaterial_{id(plane)}",
            seed=seed_mat,
        )
        plane.data.materials.append(mat)
        plane.visible_shadow = False
        plane.visible_diffuse = False
        return plane

    def finalize_assets(self, assets):
        """Snap spalling plug planes to the nearest wall plug location, then embed slightly.
        Also stores assets for later boolean application to walls (see apply_spalling_plug_boolean_to_walls).
        """
        # A tiny bit deeper than the wall plug to look like it's 'behind' it
        EMBED_OFFSET = 0.014

        # Store spalling plugs for deferred boolean application (after walls are split)
        # spalling_col = butil.get_collection("placeholders:spalling_plugs")
        # for obj in assets:
        #     butil.put_in_collection(obj, spalling_col, exclusive=False)

        for obj in assets:
            try:
                # 1. Improved name check to ensure we find the Infinigen-generated plugs
                def is_wall_plug(o):
                    n = o.name.lower()
                    return "staticwallplug" in n or any(
                        k in n for k in ("wall_socket", "socket", "outlet")
                    )

                wall_plugs = [o for o in bpy.data.objects if is_wall_plug(o)]

                if not wall_plugs:
                    logger.warning(
                        f"No wall plugs found in scene for {obj.name} to snap to."
                    )
                    continue

                # 2. Find the closest plug
                nearest = min(
                    wall_plugs,
                    key=lambda p: (p.location - obj.location).length_squared,
                )

                # 3. SNAP BOTH LOCATION AND ROTATION
                obj.location = nearest.location.copy()
                obj.rotation_euler = nearest.rotation_euler.copy()

                # 4. Push the spalling into the wall using the plug's Back face normal
                # (correct for any wall orientation - uses X/Y not Z)
                back_mask = tagged_face_mask(nearest, {t.Subpart.Back})
                if back_mask.any():
                    back_poly_idx = int(np.where(back_mask)[0][0])
                    back_poly = nearest.data.polygons[back_poly_idx]
                    into_wall_normal = np.array(
                        butil.global_polygon_normal(nearest, back_poly)
                    )
                    obj.location += into_wall_normal * EMBED_OFFSET
                else:
                    # Fallback if Back face isn't tagged: use plug's -Y (common wall axis)
                    rotation_matrix = obj.rotation_euler.to_matrix()
                    into_wall_normal = np.array(rotation_matrix @ Vector((0, -1, 0)))
                    obj.location += into_wall_normal * EMBED_OFFSET

                # Previous logic (incorrect - used local Z which changes Z offset):
                # import mathutils
                # rotation_matrix = obj.rotation_euler.to_matrix()
                # backwards_direction = rotation_matrix @ mathutils.Vector((0, 0, 1))
                # obj.location += backwards_direction * EMBED_OFFSET

                logger.info(f"Successfully snapped {obj.name} to {nearest.name}")

            except Exception as e:
                logger.warning("Failed to align spalling plug %s: %s", obj.name, e)


# Boolean/cutter logic (commented out - not needed):
# def _create_indented_only_cutter(spalling_obj, z_threshold=-0.001):
#     """
#     Create a cutter mesh from ONLY the indented faces (Z < z_threshold).
#     This ensures we cut only the spalling cavity, not the entire plane.
#     """
#     cutter = butil.copy(spalling_obj)
#     cutter.name = spalling_obj.name + "_cutter"
#
#     bm = bmesh.new()
#     bm.from_mesh(cutter.data)
#     bm.faces.ensure_lookup_table()
#
#     faces_to_delete = []
#     for face in bm.faces:
#         avg_z = sum(v.co.z for v in face.verts) / len(face.verts)
#         if avg_z >= z_threshold:
#             faces_to_delete.append(face)
#
#     bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
#     bm.to_mesh(cutter.data)
#     bm.free()
#     cutter.data.update()
#
#     if len(cutter.data.polygons) == 0:
#         logger.warning(
#             "No indented faces in spalling %s, skipping boolean", spalling_obj.name
#         )
#         cutter.name = spalling_obj.name + "_cutter_empty"
#         return cutter
#
#     sol_mod = cutter.modifiers.new(name="MakeSolid", type="SOLIDIFY")
#     sol_mod.thickness = 0.05
#     sol_mod.offset = 1.0
#
#     return cutter
#
#
# def apply_spalling_plug_boolean_to_walls(wall_objs: list, state=None):
#     """
#     Apply boolean DIFFERENCE modifier to walls using spalling plug meshes as cutters.
#     Run this after split_rooms so wall objects exist.
#     Spalling plugs are collected in placeholders:spalling_plugs during finalize_assets.
#
#     If state is provided, uses room relations to find the correct wall per spalling plug.
#     Otherwise falls back to bbox-distance. Applies Solidify on cutter and Boolean on wall.
#     """
#     try:
#         spalling_col = butil.get_collection("placeholders:spalling_plugs")
#         spalling_plugs = list(spalling_col.objects)
#     except Exception:
#         spalling_plugs = []
#
#     if not spalling_plugs or not wall_objs:
#         return
#
#     wall_by_name = {w.name: w for w in wall_objs if w.type == "MESH"}
#
#     for spalling_obj in spalling_plugs:
#         if spalling_obj.type != "MESH" or not spalling_obj.data.vertices:
#             continue
#
#         # Find the correct wall: use state room relation if available
#         best_wall = None
#         if state is not None:
#             for objkey, os in state.objs.items():
#                 if os.obj is not spalling_obj:
#                     continue
#                 for rel in os.relations:
#                     room_name = rel.target_name
#                     if (
#                         room_name in state.objs
#                         and t.Semantics.Room in state.objs[room_name].tags
#                     ):
#                         wall_name = room_name.split(".")[0] + ".wall"
#                         if wall_name in wall_by_name:
#                             best_wall = wall_by_name[wall_name]
#                             break
#                 if best_wall is not None:
#                     break
#
#         # Fallback: closest wall by bbox distance
#         if best_wall is None:
#             spalling_loc = np.array(spalling_obj.matrix_world.translation)
#             best_dist = float("inf")
#             for wall in wall_objs:
#                 if wall.type != "MESH":
#                     continue
#                 bbox_corners = [wall.matrix_world @ Vector(v) for v in wall.bound_box]
#                 bbox_min = np.array([min(c[i] for c in bbox_corners) for i in range(3)])
#                 bbox_max = np.array([max(c[i] for c in bbox_corners) for i in range(3)])
#                 d = np.maximum(bbox_min - spalling_loc, 0) + np.maximum(
#                     spalling_loc - bbox_max, 0
#                 )
#                 dist = np.linalg.norm(d)
#                 if dist < best_dist:
#                     best_dist = dist
#                     best_wall = wall
#
#         if best_wall is None:
#             logger.warning("No wall found for spalling plug %s", spalling_obj.name)
#             continue
#
#         # Create cutter from ONLY the indented faces (Z < -0.001), not the entire plane
#         cutter = _create_indented_only_cutter(spalling_obj)
#         if "_cutter_empty" in cutter.name:
#             bpy.data.objects.remove(cutter, do_unlink=True)
#             continue
#
#         # Add spalling material to wall slot so boolean-created faces get the cavity look
#         spalling_mat = (
#             spalling_obj.data.materials[0] if spalling_obj.data.materials else None
#         )
#         if spalling_mat is not None and spalling_mat.name not in (
#             m.name for m in best_wall.data.materials
#         ):
#             best_wall.data.materials.append(spalling_mat)
#
#         # Add boolean modifier to wall (wall_plug cutter: modifier not applied)
#         bool_mod = best_wall.modifiers.new(name="IndentCut", type="BOOLEAN")
#         bool_mod.operation = "DIFFERENCE"
#         bool_mod.object = cutter
#         bool_mod.solver = "EXACT"
#
#         cutter.hide_viewport = True
#         cutter.hide_render = True
#         cutter_col = butil.get_collection("placeholders:asset_cutters")
#         butil.put_in_collection(cutter, cutter_col, exclusive=False)
#
#         logger.info(
#             "Applied spalling boolean to wall %s using cutter %s",
#             best_wall.name,
#             cutter.name,
#         )
