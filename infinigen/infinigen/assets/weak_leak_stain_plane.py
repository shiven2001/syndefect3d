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


def create_weak_leak_stain_material(name: str, seed: int) -> bpy.types.Material:
    """Procedural weak water leak stain, adapted from
    defect_generation/procedural_water_leak_stain_plane_gen.py.

    Uses the same parameter ranges and randomization, but driven by FixedSeed(seed)
    so Infinigen can reproduce results deterministically per asset instance.
    """
    with FixedSeed(seed):
        mapping_scale_x = np.random.uniform(0.1, 0.5)
        mapping_scale_y = np.random.uniform(1.0, 1.5)
        mapping_scale_z = np.random.uniform(0.2, 0.3)
        mapping_loc_x = np.random.uniform(0.65, 0.75)
        mapping_loc_y = np.random.uniform(1.0, 3.0)
        noise1_scale = np.random.uniform(5.0, 10.0)
        multiply1_factor = np.random.uniform(0.4, 0.5)
        ramp2_pos0 = np.random.uniform(0.5, 0.55)
        noise2_scale = np.random.uniform(5.0, 20.0)
        # Dark brown to black for stain color (elements 1 and 2)
        stain_r = np.random.uniform(0.0, 0.2)
        stain_g = np.random.uniform(0.0, 0.1)
        stain_b = np.random.uniform(0.0, 0.05)

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
        mapping_scale_x,
        mapping_scale_y,
        mapping_scale_z,
    )
    node_mapping.inputs["Location"].default_value = (
        mapping_loc_x,
        mapping_loc_y,
        0.0,
    )
    node_mapping.inputs["Rotation"].default_value = (0.0, -90.0, 0.0)

    # Coordinate / gradient / primary noise
    links.new(node_tex_coord.outputs["Object"], node_mapping.inputs["Vector"])

    node_gradient_1 = nodes.new("ShaderNodeTexGradient")
    node_gradient_1.gradient_type = "LINEAR"
    links.new(node_mapping.outputs["Vector"], node_gradient_1.inputs["Vector"])

    node_noise_1 = nodes.new("ShaderNodeTexNoise")
    node_noise_1.noise_dimensions = "2D"
    node_noise_1.normalize = True
    node_noise_1.noise_type = "FBM"
    node_noise_1.inputs["Scale"].default_value = noise1_scale
    node_noise_1.inputs["Detail"].default_value = 2.0
    node_noise_1.inputs["Roughness"].default_value = 0.5
    links.new(node_mapping.outputs["Vector"], node_noise_1.inputs["Vector"])

    node_multiply_1 = nodes.new("ShaderNodeMath")
    node_multiply_1.operation = "MULTIPLY"
    node_multiply_1.inputs[1].default_value = multiply1_factor
    links.new(node_noise_1.outputs["Fac"], node_multiply_1.inputs[0])

    node_invert_color_1 = nodes.new("ShaderNodeInvert")
    links.new(node_gradient_1.outputs["Color"], node_invert_color_1.inputs["Color"])

    node_add_1 = nodes.new("ShaderNodeMath")
    node_add_1.operation = "ADD"
    links.new(node_invert_color_1.outputs["Color"], node_add_1.inputs[0])
    links.new(node_multiply_1.outputs["Value"], node_add_1.inputs[1])

    # Color ramp for stain color
    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements.new(0.0)
    node_ramp_1.color_ramp.elements[0].position = 0.0
    node_ramp_1.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
    node_ramp_1.color_ramp.elements[1].position = 0.5
    node_ramp_1.color_ramp.elements[1].color = (stain_r, stain_g, stain_b, 1.0)
    node_ramp_1.color_ramp.elements[2].position = 1.0
    node_ramp_1.color_ramp.elements[2].color = (stain_r, stain_g, stain_b, 1.0)
    links.new(node_add_1.outputs["Value"], node_ramp_1.inputs["Fac"])

    # Ramp controlling opacity / mask of the leak
    node_ramp_2 = nodes.new("ShaderNodeValToRGB")
    node_ramp_2.color_ramp.elements.new(0.0)
    node_ramp_2.color_ramp.elements[0].position = ramp2_pos0
    node_ramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    node_ramp_2.color_ramp.elements[1].position = 0.9
    node_ramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    node_ramp_2.color_ramp.elements[2].position = 1.0
    node_ramp_2.color_ramp.elements[2].color = (1.0, 1.0, 1.0, 1.0)
    links.new(node_add_1.outputs["Value"], node_ramp_2.inputs["Fac"])

    links.new(node_ramp_2.outputs["Color"], node_bsdf_paint.inputs["Alpha"])
    links.new(node_ramp_1.outputs["Color"], node_bsdf_paint.inputs["Base Color"])

    # Height/bump from opacity mask
    node_power_1 = nodes.new("ShaderNodeMath")
    node_power_1.operation = "POWER"
    node_power_1.inputs[1].default_value = 2.0
    links.new(node_ramp_2.outputs["Color"], node_power_1.inputs[0])

    node_multiply_height = nodes.new("ShaderNodeMath")
    node_multiply_height.operation = "MULTIPLY"
    node_multiply_height.inputs[1].default_value = 100.0
    links.new(node_power_1.outputs["Value"], node_multiply_height.inputs[0])

    node_bump_main = nodes.new("ShaderNodeBump")
    node_bump_main.inputs["Strength"].default_value = 0.2
    node_bump_main.inputs["Distance"].default_value = 2.0
    links.new(node_multiply_height.outputs["Value"], node_bump_main.inputs["Height"])
    links.new(node_bump_main.outputs["Normal"], node_bsdf_paint.inputs["Normal"])

    # Secondary noise-based bump for subtle surface texture
    node_noise_2 = nodes.new("ShaderNodeTexNoise")
    node_noise_2.noise_dimensions = "3D"
    node_noise_2.normalize = True
    node_noise_2.noise_type = "FBM"
    node_noise_2.inputs["Scale"].default_value = noise2_scale
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
    node_multiply_2.inputs[1].default_value = 0.05
    links.new(node_ramp_3.outputs["Color"], node_multiply_2.inputs[0])

    node_bump_2 = nodes.new("ShaderNodeBump")
    node_bump_2.inputs["Strength"].default_value = 1.0
    node_bump_2.inputs["Distance"].default_value = 0.5
    links.new(node_multiply_2.outputs["Value"], node_bump_2.inputs["Height"])
    links.new(node_bump_2.outputs["Normal"], node_bump_main.inputs["Normal"])

    links.new(node_bsdf_paint.outputs["BSDF"], node_output.inputs["Surface"])

    mat.blend_method = "CLIP"
    mat.shadow_method = "NONE"
    return mat


class WeakLeakStainPlaneFactory(AssetFactory):
    """Procedural wall-mounted weak water leak stain plane."""

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Same thin vertical bbox as other wall defect planes
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        plane = new_plane()

        with FixedSeed(int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))):
            scale_z_val = np.random.uniform(0.6, 1.0) * self.plane_size / 2
            scale_y_val = np.random.uniform(0.6, 1.0) * self.plane_size / 2

        plane.scale = (scale_z_val, scale_y_val, 1.0)
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)

        mat = create_weak_leak_stain_material(
            name=f"WeakLeakStainMaterial_{id(plane)}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
        )
        plane.data.materials.append(mat)
        plane.visible_shadow = False
        plane.visible_diffuse = False
        return plane

    def finalize_assets(self, assets):
        """Embed leak-stain planes slightly into the wall, same logic as paint-peel."""
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
                logger.warning(
                    "Failed to embed weak leak stain plane %s: %s", obj.name, e
                )
