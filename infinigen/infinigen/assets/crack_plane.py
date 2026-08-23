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
    """Sparse hairline plaster cracks: few long lines, mid-grey, not a black web.

    Voronoi DISTANCE_TO_EDGE draws a full cell network. A second noise mask
    keeps only a fraction of those edges so branches die out. Color stays
    close to the wall so the line is a shadow, not a marker.
    """
    with FixedSeed(seed):
        # High Voronoi scale → thin Distance-to-edge lines. A second *low-scale*
        # Voronoi keeps only a few long corridors, so the web does not fill the wall.
        hairline = np.random.rand() < 0.85
        thickness = np.random.uniform(0.005, 0.01)
        voronoi_scale = np.random.uniform(5.0, 6.0)
        voronoi_detail = np.random.uniform(0.5, 0.6)
        if hairline:
            coarse_scale = np.random.uniform(0.26, 0.48)
            coarse_width = np.random.uniform(0.016, 0.028)
            bump_strength = np.random.uniform(0.20, 0.38)
            v = np.random.uniform(0.12, 0.22)
        else:
            coarse_scale = np.random.uniform(0.22, 0.40)
            coarse_width = np.random.uniform(0.022, 0.038)
            bump_strength = np.random.uniform(0.28, 0.48)
            v = np.random.uniform(0.08, 0.18)
        voronoi_randomness = np.random.uniform(0.80, 1.0)
        noise_mix_fac = np.random.uniform(0.32, 0.48)
        noise_scale = np.random.uniform(1.0, 2.2)
        mapping_offset = (
            np.random.uniform(0, 100),
            np.random.uniform(0, 100),
            np.random.uniform(0, 100),
        )
        wall_bump_strength = np.random.uniform(0.02, 0.06)
        base_color = (v, v * 0.98, v * 0.95, 1.0)

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
    node_voronoi_1.inputs["Detail"].default_value = voronoi_detail

    links.new(node_mapping_1.outputs["Vector"], node_voronoi_1.inputs["Vector"])

    node_ramp_1 = nodes.new("ShaderNodeValToRGB")
    node_ramp_1.color_ramp.elements[0].position = 0.0
    node_ramp_1.color_ramp.elements[1].position = float(thickness)
    node_ramp_1.color_ramp.elements[0].color = (0, 0, 0, 1)
    node_ramp_1.color_ramp.elements[1].color = (1, 1, 1, 1)

    links.new(node_voronoi_1.outputs["Distance"], node_ramp_1.inputs["Fac"])

    node_bump_1 = nodes.new("ShaderNodeBump")
    node_bump_1.inputs["Strength"].default_value = bump_strength
    node_bump_1.inputs["Distance"].default_value = 0.0008

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

    # Thin web (high scale) only inside a few long corridors (low-scale Voronoi).
    node_invert_1 = nodes.new("ShaderNodeInvert")
    links.new(node_ramp_1.outputs["Color"], node_invert_1.inputs["Color"])

    node_voronoi_coarse = nodes.new("ShaderNodeTexVoronoi")
    node_voronoi_coarse.voronoi_dimensions = "3D"
    node_voronoi_coarse.feature = "DISTANCE_TO_EDGE"
    node_voronoi_coarse.inputs["Scale"].default_value = coarse_scale
    node_voronoi_coarse.inputs["Randomness"].default_value = voronoi_randomness
    node_voronoi_coarse.inputs["Roughness"].default_value = 1.0
    node_voronoi_coarse.inputs["Detail"].default_value = voronoi_detail
    links.new(node_mapping_1.outputs["Vector"], node_voronoi_coarse.inputs["Vector"])

    node_ramp_coarse = nodes.new("ShaderNodeValToRGB")
    node_ramp_coarse.color_ramp.elements[0].position = 0.0
    node_ramp_coarse.color_ramp.elements[1].position = float(coarse_width)
    node_ramp_coarse.color_ramp.elements[0].color = (0, 0, 0, 1)
    node_ramp_coarse.color_ramp.elements[1].color = (1, 1, 1, 1)
    links.new(node_voronoi_coarse.outputs["Distance"], node_ramp_coarse.inputs["Fac"])

    node_invert_coarse = nodes.new("ShaderNodeInvert")
    links.new(node_ramp_coarse.outputs["Color"], node_invert_coarse.inputs["Color"])

    node_alpha = nodes.new("ShaderNodeMath")
    node_alpha.operation = "MULTIPLY"
    node_alpha.use_clamp = True
    links.new(node_invert_1.outputs["Color"], node_alpha.inputs[0])
    links.new(node_invert_coarse.outputs["Color"], node_alpha.inputs[1])
    links.new(node_alpha.outputs["Value"], node_bsdf_paint.inputs["Alpha"])

    links.new(node_bsdf_paint.outputs["BSDF"], node_output.inputs["Surface"])

    mat.blend_method = "CLIP"
    if hasattr(mat, "shadow_method"):
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


class CeilingCrackPlaneFactory(AssetFactory):
    """Same crack material as walls, hung flush to the ceiling (Top vs ceiling)."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
        # Horizontal thin box: Top/Bottom are the large faces for hanging.
        ph = new_bbox(-0.5, 0.5, -0.5, 0.5, -0.005, 0.005)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        plane = new_plane()
        with FixedSeed(int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))):
            sx = np.random.uniform(0.6, 1.0) * self.plane_size / 2
            sy = np.random.uniform(0.6, 1.0) * self.plane_size / 2
        plane.scale = (sx, sy, 1)
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
        EMBED_OFFSET = -0.000226
        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                from infinigen.core import tags as t
                from infinigen.core.tagging import tagged_face_mask

                top_mask = tagged_face_mask(obj, {t.Subpart.Top})
                if top_mask.any():
                    top_faces = [i for i, tag in enumerate(top_mask) if tag]
                    top_poly = obj.data.polygons[
                        max(top_faces, key=lambda idx: obj.data.polygons[idx].area)
                    ]
                else:
                    top_poly = max(
                        obj.data.polygons,
                        key=lambda p: p.normal.z,
                    )
                ceil_normal = np.array(butil.global_polygon_normal(obj, top_poly))
                obj.location += Vector(ceil_normal * EMBED_OFFSET)
            except Exception as e:
                logger.warning(
                    "Failed to embed ceiling crack plane %s: %s", obj.name, e
                )
