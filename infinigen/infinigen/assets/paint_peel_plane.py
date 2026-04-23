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


def create_paint_peel_material(name: str, seed: int) -> bpy.types.Material:
    """Procedural paint peeling with randomized parameters (4D noise, bump, wall texture).
    Adapted from defect_generation/procedural_paint_peel_plane_gen.py."""
    with FixedSeed(seed):
        noise_1_w = np.random.uniform(1.0, 10.0)
        noise_1_scale = np.random.uniform(1.0, 15.0)
        ramp_1_pos_0 = np.random.uniform(0.42, 0.45)
        ramp_1_pos_1 = ramp_1_pos_0 + np.random.uniform(0.01, 0.03)
        bump_1_strength = np.random.uniform(0.5, 1.0)
        noise_2_scale = np.random.uniform(10.0, 60.0)
        multiply_2_val = np.random.uniform(0.02, 0.03)
        base_color_val = np.random.uniform(0.90, 1.0)  # white to light grey

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_output = nodes.new("ShaderNodeOutputMaterial")
    node_tex_coord = nodes.new("ShaderNodeTexCoord")
    node_noise_1 = nodes.new("ShaderNodeTexNoise")

    node_noise_1.noise_dimensions = "4D"
    node_noise_1.normalize = False
    node_noise_1.noise_type = "FBM"
    node_noise_1.inputs["W"].default_value = noise_1_w
    node_noise_1.inputs["Scale"].default_value = noise_1_scale
    node_noise_1.inputs["Detail"].default_value = 20.0
    node_noise_1.inputs["Roughness"].default_value = 0.5

    links.new(node_tex_coord.outputs["Object"], node_noise_1.inputs["Vector"])

    node_power_1 = nodes.new("ShaderNodeMath")
    node_power_1.operation = "POWER"
    node_power_1.inputs[1].default_value = 0.8

    links.new(node_noise_1.outputs["Color"], node_power_1.inputs[0])

    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements.new(1.000)
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

    # Wall texture
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
    node_bsdf_paint.inputs["Base Color"].default_value = (
        base_color_val,
        base_color_val,
        base_color_val,
        1.0,
    )

    links.new(node_bump_1.outputs["Normal"], node_bsdf_paint.inputs["Normal"])
    links.new(node_bsdf_paint.outputs["BSDF"], node_output.inputs["Surface"])
    links.new(node_ramp_1.outputs["Color"], node_bsdf_paint.inputs["Alpha"])

    mat.blend_method = "CLIP"
    mat.shadow_method = "CLIP"
    return mat


class PaintPeelPlaneFactory(AssetFactory):
    """Procedural wall-mounted paint peeling plane (centered, sparse, non-metallic)."""

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Same geometry/orientation as crack planes: thin vertical bbox
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

        mat = create_paint_peel_material(
            name=f"PaintPeelMaterial_{id(plane)}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
        )
        plane.data.materials.append(mat)
        plane.visible_shadow = False
        plane.visible_diffuse = False
        return plane

    def finalize_assets(self, assets):
        """Embed paint-peel planes slightly into the wall, same logic as crack planes."""
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
                logger.warning("Failed to embed paint peel plane %s: %s", obj.name, e)
