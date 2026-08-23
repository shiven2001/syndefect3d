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

_SCENE_SUBSTRATE = None
_SCENE_SUBSTRATE_KEY = None


def _generation_seed():
    try:
        import gin

        return int(gin.query_parameter("OVERALL_SEED"))
    except Exception:
        return 0


def sample_peel_substrate(seed: int):
    """Exposed plaster/concrete/paint under the film. One draw per apartment."""
    with FixedSeed(seed):
        kind = np.random.choice(
            ["tan", "concrete", "white", "beige", "cool_grey"],
            p=[0.22, 0.30, 0.22, 0.16, 0.10],
        )
        if kind == "tan":
            plaster_lo = (
                np.random.uniform(0.34, 0.42),
                np.random.uniform(0.32, 0.39),
                np.random.uniform(0.26, 0.34),
                1.0,
            )
            plaster_hi = (
                np.random.uniform(0.50, 0.58),
                np.random.uniform(0.46, 0.54),
                np.random.uniform(0.40, 0.48),
                1.0,
            )
        elif kind == "concrete":
            g0 = np.random.uniform(0.36, 0.46)
            g1 = np.random.uniform(0.56, 0.68)
            plaster_lo = (g0 * 0.98, g0, g0 * 1.02, 1.0)
            plaster_hi = (g1 * 0.99, g1, g1 * 1.01, 1.0)
        elif kind == "white":
            g0 = np.random.uniform(0.68, 0.78)
            g1 = np.random.uniform(0.86, 0.95)
            plaster_lo = (g0, g0 * 0.995, g0 * 0.98, 1.0)
            plaster_hi = (g1, g1 * 0.998, g1 * 0.99, 1.0)
        elif kind == "beige":
            plaster_lo = (
                np.random.uniform(0.52, 0.62),
                np.random.uniform(0.48, 0.58),
                np.random.uniform(0.40, 0.50),
                1.0,
            )
            plaster_hi = (
                np.random.uniform(0.72, 0.82),
                np.random.uniform(0.68, 0.78),
                np.random.uniform(0.58, 0.68),
                1.0,
            )
        else:
            g0 = np.random.uniform(0.40, 0.50)
            g1 = np.random.uniform(0.60, 0.72)
            plaster_lo = (g0 * 0.96, g0, g0 * 1.04, 1.0)
            plaster_hi = (g1 * 0.97, g1, g1 * 1.03, 1.0)
        pit_tint = tuple(min(c * 0.55, 1.0) for c in plaster_lo[:3]) + (1.0,)
        flake_v = np.random.uniform(0.86, 0.96)
        flake_color = (flake_v, flake_v * 0.99, flake_v * 0.97, 1.0)
    return plaster_lo, plaster_hi, pit_tint, flake_color


def scene_peel_substrate():
    """Same exposed color for every peel in this apartment / generation seed."""
    global _SCENE_SUBSTRATE, _SCENE_SUBSTRATE_KEY
    key = _generation_seed()
    if _SCENE_SUBSTRATE is not None and _SCENE_SUBSTRATE_KEY == key:
        return _SCENE_SUBSTRATE
    _SCENE_SUBSTRATE = sample_peel_substrate(
        int_hash((key, "paint_peel_substrate"))
    )
    _SCENE_SUBSTRATE_KEY = key
    return _SCENE_SUBSTRATE


def create_paint_peel_material(name: str, seed: int) -> bpy.types.Material:
    """Paint peel as a two-layer film: intact paint is transparent (real wall
    shows through); peeled patches are a grainy plaster substrate with a thin
    lifted lip. Substrate tint is scene-wide; patch shape is per instance.
    """
    plaster_lo, plaster_hi, pit_tint, flake_color = scene_peel_substrate()
    with FixedSeed(seed):
        patch_w = np.random.uniform(0.4, 3.2)
        patch_scale = np.random.uniform(1.4, 3.2)
        chip_scale = np.random.uniform(7.0, 16.0)
        chip_amt = np.random.uniform(0.18, 0.38)
        ramp0 = np.random.uniform(0.40, 0.46)
        ramp1 = min(ramp0 + np.random.uniform(0.02, 0.045), 0.92)
        grain_scale = np.random.uniform(70.0, 140.0)
        body_scale = np.random.uniform(14.0, 28.0)
        pit_scale = np.random.uniform(28.0, 55.0)
        # Same invert-bump rim as the original peel (Distance ~20).
        rim_strength = np.random.uniform(0.65, 1.15)
        rim_distance = np.random.uniform(14.0, 22.0)
        plaster_bump = np.random.uniform(0.35, 0.55)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_output = nodes.new("ShaderNodeOutputMaterial")
    tex = nodes.new("ShaderNodeTexCoord")

    # --- Peel mask: large irregular patches plus smaller chips / islands ---
    n_patch = nodes.new("ShaderNodeTexNoise")
    n_patch.noise_dimensions = "4D"
    n_patch.normalize = False
    n_patch.noise_type = "FBM"
    n_patch.inputs["W"].default_value = patch_w
    n_patch.inputs["Scale"].default_value = patch_scale
    n_patch.inputs["Detail"].default_value = 16.0
    n_patch.inputs["Roughness"].default_value = 0.62
    links.new(tex.outputs["Object"], n_patch.inputs["Vector"])

    n_chip = nodes.new("ShaderNodeTexNoise")
    n_chip.noise_dimensions = "3D"
    n_chip.normalize = True
    n_chip.noise_type = "FBM"
    n_chip.inputs["Scale"].default_value = chip_scale
    n_chip.inputs["Detail"].default_value = 8.0
    n_chip.inputs["Roughness"].default_value = 0.7
    links.new(tex.outputs["Object"], n_chip.inputs["Vector"])

    mix_mask = nodes.new("ShaderNodeMixRGB")
    mix_mask.blend_type = "ADD"
    mix_mask.use_clamp = True
    mix_mask.inputs["Fac"].default_value = chip_amt
    links.new(n_patch.outputs["Fac"], mix_mask.inputs["Color1"])
    links.new(n_chip.outputs["Fac"], mix_mask.inputs["Color2"])

    peel_ramp = nodes.new("ShaderNodeValToRGB")
    peel_ramp.color_ramp.elements.new(1.0)
    peel_ramp.color_ramp.interpolation = "EASE"
    peel_ramp.color_ramp.elements[0].position = ramp0
    peel_ramp.color_ramp.elements[1].position = ramp1
    peel_ramp.color_ramp.elements[2].position = 1.0
    peel_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    peel_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    peel_ramp.color_ramp.elements[2].color = (1, 1, 1, 1)
    links.new(mix_mask.outputs["Color"], peel_ramp.inputs["Fac"])

    # Lip / curl ring: peaks at the paint–substrate boundary.
    one_minus = nodes.new("ShaderNodeMath")
    one_minus.operation = "SUBTRACT"
    one_minus.inputs[0].default_value = 1.0
    links.new(peel_ramp.outputs["Color"], one_minus.inputs[1])

    edge = nodes.new("ShaderNodeMath")
    edge.operation = "MULTIPLY"
    links.new(peel_ramp.outputs["Color"], edge.inputs[0])
    links.new(one_minus.outputs["Value"], edge.inputs[1])

    edge_amp = nodes.new("ShaderNodeMath")
    edge_amp.operation = "MULTIPLY"
    edge_amp.inputs[1].default_value = 4.0
    links.new(edge.outputs["Value"], edge_amp.inputs[0])

    # --- Substrate: sandy plaster (body + fine grain + pits) ---
    n_body = nodes.new("ShaderNodeTexNoise")
    n_body.noise_dimensions = "3D"
    n_body.normalize = True
    n_body.noise_type = "FBM"
    n_body.inputs["Scale"].default_value = body_scale
    n_body.inputs["Detail"].default_value = 14.0
    n_body.inputs["Roughness"].default_value = 0.68
    links.new(tex.outputs["Object"], n_body.inputs["Vector"])

    n_grain = nodes.new("ShaderNodeTexNoise")
    n_grain.noise_dimensions = "3D"
    n_grain.normalize = True
    n_grain.noise_type = "FBM"
    n_grain.inputs["Scale"].default_value = grain_scale
    n_grain.inputs["Detail"].default_value = 12.0
    n_grain.inputs["Roughness"].default_value = 0.8
    links.new(tex.outputs["Object"], n_grain.inputs["Vector"])

    n_pits = nodes.new("ShaderNodeTexVoronoi")
    n_pits.voronoi_dimensions = "3D"
    n_pits.feature = "F1"
    n_pits.inputs["Scale"].default_value = pit_scale
    n_pits.inputs["Randomness"].default_value = 1.0
    links.new(tex.outputs["Object"], n_pits.inputs["Vector"])

    plaster_ramp = nodes.new("ShaderNodeValToRGB")
    plaster_ramp.color_ramp.elements[0].position = 0.15
    plaster_ramp.color_ramp.elements[1].position = 0.85
    plaster_ramp.color_ramp.elements[0].color = plaster_lo
    plaster_ramp.color_ramp.elements[1].color = plaster_hi
    links.new(n_body.outputs["Fac"], plaster_ramp.inputs["Fac"])

    grain_mix = nodes.new("ShaderNodeMixRGB")
    grain_mix.blend_type = "MIX"
    grain_mix.inputs["Fac"].default_value = 0.12
    links.new(plaster_ramp.outputs["Color"], grain_mix.inputs["Color1"])
    links.new(n_grain.outputs["Color"], grain_mix.inputs["Color2"])

    pit_ramp = nodes.new("ShaderNodeValToRGB")
    pit_ramp.color_ramp.elements[0].position = 0.0
    pit_ramp.color_ramp.elements[1].position = 0.45
    pit_ramp.color_ramp.elements[0].color = pit_tint
    pit_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    links.new(n_pits.outputs["Distance"], pit_ramp.inputs["Fac"])

    plaster_col = nodes.new("ShaderNodeMixRGB")
    plaster_col.blend_type = "MULTIPLY"
    plaster_col.inputs["Fac"].default_value = 0.22
    links.new(grain_mix.outputs["Color"], plaster_col.inputs["Color1"])
    links.new(pit_ramp.outputs["Color"], plaster_col.inputs["Color2"])

    # Soft crease under the flake (grey, not black).
    crease = nodes.new("ShaderNodeMixRGB")
    crease.blend_type = "MULTIPLY"
    crease.inputs["Fac"].default_value = 0.18
    crease.inputs["Color2"].default_value = (0.42, 0.40, 0.37, 1.0)
    links.new(plaster_col.outputs["Color"], crease.inputs["Color1"])
    links.new(edge_amp.outputs["Value"], crease.inputs["Fac"])

    paint_mix = nodes.new("ShaderNodeMixRGB")
    paint_mix.blend_type = "MIX"
    paint_mix.inputs["Color2"].default_value = flake_color
    links.new(edge_amp.outputs["Value"], paint_mix.inputs["Fac"])
    links.new(crease.outputs["Color"], paint_mix.inputs["Color1"])

    # Height: plaster grain, then a millimetre-scale raised lip (not Distance=20).
    grain_h = nodes.new("ShaderNodeMath")
    grain_h.operation = "MULTIPLY"
    grain_h.inputs[1].default_value = 0.45
    links.new(n_grain.outputs["Fac"], grain_h.inputs[0])

    body_h = nodes.new("ShaderNodeMath")
    body_h.operation = "MULTIPLY"
    body_h.inputs[1].default_value = 0.25
    links.new(n_body.outputs["Fac"], body_h.inputs[0])

    pit_h = nodes.new("ShaderNodeMath")
    pit_h.operation = "MULTIPLY"
    pit_h.inputs[1].default_value = 0.2
    links.new(n_pits.outputs["Distance"], pit_h.inputs[0])

    h1 = nodes.new("ShaderNodeMath")
    h1.operation = "ADD"
    links.new(grain_h.outputs["Value"], h1.inputs[0])
    links.new(body_h.outputs["Value"], h1.inputs[1])

    h2 = nodes.new("ShaderNodeMath")
    h2.operation = "ADD"
    links.new(h1.outputs["Value"], h2.inputs[0])
    links.new(pit_h.outputs["Value"], h2.inputs[1])

    bump_plaster = nodes.new("ShaderNodeBump")
    bump_plaster.inputs["Strength"].default_value = plaster_bump
    bump_plaster.inputs["Distance"].default_value = 0.0025
    links.new(h2.outputs["Value"], bump_plaster.inputs["Height"])

    # Original peel rim: sharp invert bump on the mask (power 5, *2, Distance ~20).
    rim_pow = nodes.new("ShaderNodeMath")
    rim_pow.operation = "POWER"
    rim_pow.inputs[1].default_value = 5.0
    links.new(peel_ramp.outputs["Color"], rim_pow.inputs[0])

    rim_mul = nodes.new("ShaderNodeMath")
    rim_mul.operation = "MULTIPLY"
    rim_mul.inputs[1].default_value = 2.0
    links.new(rim_pow.outputs["Value"], rim_mul.inputs[0])

    bump_lip = nodes.new("ShaderNodeBump")
    bump_lip.invert = True
    bump_lip.inputs["Strength"].default_value = rim_strength
    bump_lip.inputs["Distance"].default_value = rim_distance
    links.new(rim_mul.outputs["Value"], bump_lip.inputs["Height"])
    links.new(bump_plaster.outputs["Normal"], bump_lip.inputs["Normal"])

    rough = nodes.new("ShaderNodeMath")
    rough.operation = "MULTIPLY"
    rough.use_clamp = True
    rough.inputs[1].default_value = 0.35
    links.new(edge_amp.outputs["Value"], rough.inputs[0])

    rough_sub = nodes.new("ShaderNodeMath")
    rough_sub.operation = "SUBTRACT"
    rough_sub.use_clamp = True
    rough_sub.inputs[0].default_value = 0.92
    links.new(rough.outputs["Value"], rough_sub.inputs[1])

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(paint_mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough_sub.outputs["Value"], bsdf.inputs["Roughness"])
    links.new(bump_lip.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(peel_ramp.outputs["Color"], bsdf.inputs["Alpha"])
    spec_key = (
        "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    )
    if spec_key in bsdf.inputs:
        bsdf.inputs[spec_key].default_value = 0.12
    links.new(bsdf.outputs["BSDF"], node_output.inputs["Surface"])

    mat.blend_method = "CLIP"
    if hasattr(mat, "shadow_method"):
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


class CeilingPeelFactory(AssetFactory):
    """Same peel material as walls, hung flush to the ceiling (Top vs ceiling).

    Name is short on purpose: Blender object names cap at 63 chars, and
    ``CeilingPaintPeelPlaneFactory(...).spawn_placeholder(...)`` was truncated
    so populate could not parse the instance seed.
    """

    def __init__(self, factory_seed, coarse: bool = False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.5, 1.5)

    def create_placeholder(self, **kwargs):
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
        mat = create_paint_peel_material(
            name=f"PaintPeelMaterial_{id(plane)}",
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
                    top_poly = max(obj.data.polygons, key=lambda p: p.normal.z)
                ceil_normal = np.array(butil.global_polygon_normal(obj, top_poly))
                obj.location += Vector(ceil_normal * EMBED_OFFSET)
            except Exception as e:
                logger.warning(
                    "Failed to embed ceiling paint peel plane %s: %s", obj.name, e
                )
