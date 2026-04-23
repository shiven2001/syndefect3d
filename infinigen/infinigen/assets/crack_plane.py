# Copyright (C) 2023, Princeton University.
# Procedural crack plane asset for Infinigen (wall-mounted, defect semantics).
# Adapted from defect_generation/procedural_crack_plane_gen.py.

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


def create_crack_material(name: str, seed: int) -> bpy.types.Material:
    """Create a procedural hairline-crack material (transparent mesh, dark cracks visible).
    Adapted from defect_generation/procedural_crack_plane_gen.py create_cracks().
    """
    with FixedSeed(seed):
        thickness = np.random.uniform(0.001, 0.003)
        voronoi_scale = np.random.uniform(0.45, 2.0)
        voronoi_randomness = np.random.uniform(0.75, 1.0)
        noise_mix_fac = np.random.uniform(0.35, 0.5)
        noise_scale = np.random.uniform(1.0, 2.5)
        mapping_offset = (
            np.random.uniform(0, 100),
            np.random.uniform(0, 100),
            np.random.uniform(0, 100),
        )
        bump_strength = np.random.uniform(0.5, 0.8)
        wall_bump_strength = np.random.uniform(0.02, 0.08)
        # Base color: lighter grey #DCDCDC to #9E9E9E (RGB 0.62–0.86)
        v = np.random.uniform(158 / 255, 220 / 255)  # #9E9E9E to #DCDCDC
        base_color = (v, v, v, 1.0)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_output = nodes.new("ShaderNodeOutputMaterial")
    node_tex_coord = nodes.new("ShaderNodeTexCoord")
    node_noise_1 = nodes.new("ShaderNodeTexNoise")

    node_noise_1.noise_dimensions = "3D"
    node_noise_1.normalize = True
    node_noise_1.noise_type = "FBM"
    node_noise_1.inputs["Scale"].default_value = noise_scale
    node_noise_1.inputs["Detail"].default_value = 5.0
    node_noise_1.inputs["Roughness"].default_value = 0.5

    links.new(node_tex_coord.outputs["Object"], node_noise_1.inputs["Vector"])

    node_color_mix_1 = nodes.new("ShaderNodeMixRGB")
    node_color_mix_1.blend_type = "MIX"
    node_color_mix_1.use_clamp = True
    node_color_mix_1.inputs["Fac"].default_value = noise_mix_fac

    links.new(node_noise_1.outputs["Color"], node_color_mix_1.inputs["Color1"])
    links.new(node_tex_coord.outputs["Object"], node_color_mix_1.inputs["Color2"])

    node_mapping_1 = nodes.new("ShaderNodeMapping")
    node_mapping_1.inputs["Location"].default_value[0] = mapping_offset[0]
    node_mapping_1.inputs["Location"].default_value[1] = mapping_offset[1]
    node_mapping_1.inputs["Location"].default_value[2] = mapping_offset[2]
    links.new(node_color_mix_1.outputs["Color"], node_mapping_1.inputs["Vector"])

    node_voronoi_1 = nodes.new("ShaderNodeTexVoronoi")
    node_voronoi_1.voronoi_dimensions = "3D"
    node_voronoi_1.feature = "DISTANCE_TO_EDGE"
    node_voronoi_1.inputs["Scale"].default_value = voronoi_scale
    node_voronoi_1.inputs["Randomness"].default_value = voronoi_randomness
    node_voronoi_1.inputs["Roughness"].default_value = 1.0
    node_voronoi_1.inputs["Detail"].default_value = 0.001

    links.new(node_mapping_1.outputs["Vector"], node_voronoi_1.inputs["Vector"])

    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements[0].position = 0.0
    node_ramp_1.color_ramp.elements[1].position = thickness

    links.new(node_voronoi_1.outputs["Distance"], node_ramp_1.inputs["Fac"])

    node_bump_1 = nodes.new("ShaderNodeBump")
    node_bump_1.inputs["Strength"].default_value = bump_strength
    node_bump_1.inputs["Distance"].default_value = 1.0

    links.new(node_ramp_1.outputs["Color"], node_bump_1.inputs["Height"])

    node_bsdf_paint = nodes.new("ShaderNodeBsdfPrincipled")
    node_bsdf_paint.inputs["Base Color"].default_value = base_color
    links.new(node_bump_1.outputs["Normal"], node_bsdf_paint.inputs["Normal"])

    # Wall paint surface texture (noise-based bump)
    noise_texture_2 = nodes.new("ShaderNodeTexNoise")
    noise_texture_2.noise_dimensions = "3D"
    noise_texture_2.normalize = True
    noise_texture_2.noise_type = "FBM"
    noise_texture_2.inputs["Scale"].default_value = 100.0
    noise_texture_2.inputs["Detail"].default_value = 2.0
    noise_texture_2.inputs["Roughness"].default_value = 0.5

    node_ramp_2 = nodes.new("ShaderNodeValToRGB")
    links.new(noise_texture_2.outputs["Color"], node_ramp_2.inputs["Fac"])

    node_multiply_1 = nodes.new("ShaderNodeMath")
    node_multiply_1.operation = "MULTIPLY"
    node_multiply_1.inputs[1].default_value = wall_bump_strength

    links.new(node_ramp_2.outputs["Color"], node_multiply_1.inputs[0])

    node_bump_2 = nodes.new("ShaderNodeBump")
    node_bump_2.inputs["Strength"].default_value = 1.0
    node_bump_2.inputs["Distance"].default_value = 1.0

    links.new(node_multiply_1.outputs["Value"], node_bump_2.inputs["Height"])
    links.new(node_bump_2.outputs["Normal"], node_bump_1.inputs["Normal"])

    # Alpha: invert crack ramp so cracks are opaque, wall is transparent
    node_invert_1 = nodes.new("ShaderNodeInvert")
    links.new(node_ramp_1.outputs["Color"], node_invert_1.inputs["Color"])
    links.new(node_invert_1.outputs["Color"], node_bsdf_paint.inputs["Alpha"])

    links.new(node_bsdf_paint.outputs["BSDF"], node_output.inputs["Surface"])

    mat.blend_method = "CLIP"
    mat.shadow_method = "NONE"
    return mat


class CrackPlaneFactory(AssetFactory):
    """Procedural wall-mounted crack plane (hairline cracks). Uses same placement and scoring as defect planes."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        # Randomize max extent; push range up so cracks are not too tiny
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Vertical thin box: thin in X (Back/Front), extent in Y/Z for wall-mounted plaque
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        # tag_canonical_surfaces expects triangulated mesh (vert_mask_to_tri_mask uses 3 verts per poly)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        # Required for constraint solver: Back/Front/Top/Bottom face tags for StableAgainst(back, wall)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        plane = new_plane()

        # Per-instance geometric variation (size only; no random rotation)
        with FixedSeed(int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))):
            # To affect World Z (height) after a 90-deg Y rotation, we scale Local X.
            scale_z_val = (
                np.random.uniform(0.6, 1.0) * self.plane_size / 2
            )  # Becomes World Z
            scale_y_val = (
                np.random.uniform(0.6, 1.0) * self.plane_size / 2
            )  # Stays World Y

        # 1. Scale first (Local X and Y are the surface of the plane)
        plane.scale = (scale_z_val, scale_y_val, 1)

        # 2. Rotate to face the wall (normal along +X), no additional random roll
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)

        # 3. Apply transform (keeps geometry, resets transform)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)
        mat = create_crack_material(
            name=f"CrackMaterial_{id(plane)}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
        )
        plane.data.materials.append(mat)
        plane.visible_shadow = False
        plane.visible_diffuse = False
        return plane

    def finalize_assets(self, assets):
        """Embed crack planes into the wall (same as StaticDefectPlaneFactory)."""
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
                logger.warning("Failed to embed crack plane %s: %s", obj.name, e)
