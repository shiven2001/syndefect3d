# Copyright (C) 2023, Princeton University.
# Procedural open wiring plane asset for Infinigen (wall-mounted, defect semantics).
# Adapted from defect_generation/procedural_open_wiring_plane_gen.py.

import logging

import bpy
import numpy as np
from mathutils import Vector

from infinigen.assets.utils.object import new_bbox, new_plane
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.tagging import tag_canonical_surfaces
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash

logger = logging.getLogger(__name__)

# Scale factor: infinigen plane is 1x1 (half=0.5), procedural uses 6x6 (half=3)
_PLANE_SCALE = 0.5 / 3.0


def create_open_wiring_material(name: str, seed: int) -> bpy.types.Material:
    """Create plane with procedural hole material (spherical gradient).
    Adapted from defect_generation/procedural_open_wiring_plane_gen.py create_plane_with_hole_material.
    Same randomizations: noise scales, ramp positions, wall texture.
    """
    with FixedSeed(seed):
        noise_1_scale = np.random.uniform(5.0, 10.0)
        node_value_val = np.random.uniform(7.0, 12.0)
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
    links.new(node_tex.outputs["Object"], node_noise_1.inputs["Vector"])

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


def create_wire_curves(
    plane_obj: bpy.types.Object,
    seed: int,
    scale_factor: float = _PLANE_SCALE,
) -> list:
    """Create wire curves parented to plane. Same randomizations as procedural script."""
    colors = [
        (0.2, 0.01, 0.01, 1),
        (0.01, 0.01, 0.2, 1),
        (0.01, 0.15, 0.03, 1),
        (0.25, 0.05, 0, 1),
        (0.2, 0.15, 0, 1),
    ]

    with FixedSeed(seed):
        num_wires = np.random.randint(2, 5)

    wire_objs = []
    for i in range(num_wires):
        with FixedSeed(int_hash((seed, i, "wire"))):
            rgba = tuple(colors[np.random.randint(0, len(colors))])
            wire_mat = bpy.data.materials.new(name=f"Wire_Mat_{seed}_{i}")
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

            curve_data = bpy.data.curves.new(
                name=f"Wire_Curve_{seed}_{i}", type="CURVE"
            )
            curve_data.dimensions = "3D"
            curve_data.fill_mode = "FULL"
            curve_data.bevel_depth = np.random.uniform(0.025, 0.045) * scale_factor
            curve_data.bevel_resolution = 6

            wire_obj = bpy.data.objects.new(f"Wire_Obj_{seed}_{i}", curve_data)
            bpy.context.collection.objects.link(wire_obj)
            wire_obj.data.materials.append(wire_mat)

            spline = curve_data.splines.new("BEZIER")
            num_points = 5
            spline.bezier_points.add(num_points - 1)

            last_pos = Vector((0, 0, 0))
            h_size = 0.4 * scale_factor

            for p_idx in range(num_points):
                p = spline.bezier_points[p_idx]
                if p_idx == 0:
                    p.co = last_pos
                else:
                    offset = Vector(
                        (
                            np.random.uniform(-0.6, 0.6) * scale_factor,
                            np.random.uniform(-0.6, 0.6) * scale_factor,
                            np.random.uniform(0.4, 0.7) * scale_factor,
                        )
                    )
                    p.co = last_pos + offset
                    last_pos = p.co

                p.handle_left = p.co + Vector(
                    (
                        np.random.uniform(-h_size, h_size),
                        np.random.uniform(-h_size, h_size),
                        np.random.uniform(-h_size, h_size),
                    )
                )
                p.handle_right = p.co + Vector(
                    (
                        np.random.uniform(-h_size, h_size),
                        np.random.uniform(-h_size, h_size),
                        np.random.uniform(-h_size, h_size),
                    )
                )

            wire_obj.parent = plane_obj
            wire_objs.append(wire_obj)

    return wire_objs


class OpenWiringPlaneFactory(AssetFactory):
    """Procedural wall-mounted open wiring plane (hole with exposed wires)."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
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

        mat_seed = int_hash((self.factory_seed, kwargs.get("i", 0)))
        mat = create_open_wiring_material(
            name=f"OpenWiringMaterial_{id(plane)}",
            seed=mat_seed,
        )
        plane.data.materials.append(mat)

        scale_factor = min(scale_z_val, scale_y_val) * _PLANE_SCALE
        create_wire_curves(
            plane,
            seed=int_hash((self.factory_seed, kwargs.get("i", 0), "wires")),
            scale_factor=scale_factor,
        )

        plane.visible_shadow = False
        plane.visible_diffuse = False
        return plane

    def finalize_assets(self, assets):
        """Embed open wiring planes slightly into the wall."""
        EMBED_OFFSET = -0.000226
        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                from infinigen.core import tags as t
                from infinigen.core.tagging import tagged_face_mask

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
                translation = Vector(wall_normal * EMBED_OFFSET)
                obj.location += translation
            except Exception as e:
                logger.warning("Failed to embed open wiring plane %s: %s", obj.name, e)
