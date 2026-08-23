# Procedural paint-patchiness / touch-up mismatch for Infinigen.
# Flat wall decal: same paint as the wall, slightly off in color and sheen,
# with a feathered brush-edge blob (not a raised card).

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


def _set_blend(mat: bpy.types.Material):
    mat.blend_method = "BLEND"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False


def _build_patch_mask(nt, seed: int):
    """Guaranteed central blob on the YZ plane, feathered + brush wisps."""
    nodes, links = nt.nodes, nt.links
    rng = np.random.default_rng(seed)

    tex = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tex.outputs["Object"], sep.inputs["Vector"])

    # After Y-rot apply, the decal lives in YZ. Keep the window on those axes.
    comb = nodes.new("ShaderNodeCombineXYZ")
    links.new(sep.outputs["Y"], comb.inputs["X"])
    links.new(sep.outputs["Z"], comb.inputs["Y"])

    warp_n = nodes.new("ShaderNodeTexNoise")
    warp_n.noise_dimensions = "3D"
    warp_n.inputs["Scale"].default_value = float(rng.uniform(3.0, 5.5))
    warp_n.inputs["Detail"].default_value = 2.0
    warp_n.inputs["Roughness"].default_value = 0.45
    links.new(tex.outputs["Object"], warp_n.inputs["Vector"])
    warp_scale = nodes.new("ShaderNodeVectorMath")
    warp_scale.operation = "MULTIPLY"
    warp_scale.inputs[1].default_value = (
        float(rng.uniform(0.02, 0.045)),
        float(rng.uniform(0.02, 0.045)),
        0.0,
    )
    links.new(warp_n.outputs["Color"], warp_scale.inputs[0])
    warped = nodes.new("ShaderNodeVectorMath")
    warped.operation = "ADD"
    links.new(comb.outputs["Vector"], warped.inputs[0])
    links.new(warp_scale.outputs["Vector"], warped.inputs[1])

    # Stretch into a slightly oval touch-up.
    stretch = nodes.new("ShaderNodeVectorMath")
    stretch.operation = "DIVIDE"
    sx = float(rng.uniform(0.10, 0.16))
    sy = float(rng.uniform(0.08, 0.14))
    stretch.inputs[1].default_value = (sx, sy, 1.0)
    links.new(warped.outputs["Vector"], stretch.inputs[0])

    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    links.new(stretch.outputs["Vector"], length.inputs[0])

    radial = nodes.new("ShaderNodeMapRange")
    radial.inputs["From Min"].default_value = 0.55
    radial.inputs["From Max"].default_value = 1.15
    radial.inputs["To Min"].default_value = 1.0
    radial.inputs["To Max"].default_value = 0.0
    if hasattr(radial, "clamp"):
        radial.clamp = True
    links.new(length.outputs["Value"], radial.inputs["Value"])

    stroke_map = nodes.new("ShaderNodeMapping")
    stroke_map.inputs["Rotation"].default_value[2] = float(rng.uniform(-0.35, 0.35))
    stroke_map.inputs["Scale"].default_value = (
        float(rng.uniform(10.0, 18.0)),
        float(rng.uniform(1.8, 3.0)),
        1.0,
    )
    links.new(warped.outputs["Vector"], stroke_map.inputs["Vector"])
    stroke = nodes.new("ShaderNodeTexNoise")
    stroke.noise_dimensions = "3D"
    stroke.inputs["Scale"].default_value = float(rng.uniform(4.0, 7.0))
    stroke.inputs["Detail"].default_value = 5.0
    stroke.inputs["Roughness"].default_value = 0.55
    links.new(stroke_map.outputs["Vector"], stroke.inputs["Vector"])

    one_minus = nodes.new("ShaderNodeMath")
    one_minus.operation = "SUBTRACT"
    one_minus.inputs[0].default_value = 1.0
    links.new(radial.outputs["Result"], one_minus.inputs[1])
    rim = nodes.new("ShaderNodeMath")
    rim.operation = "MULTIPLY"
    rim.use_clamp = True
    links.new(radial.outputs["Result"], rim.inputs[0])
    links.new(one_minus.outputs["Value"], rim.inputs[1])
    wisp = nodes.new("ShaderNodeMath")
    wisp.operation = "MULTIPLY"
    wisp.use_clamp = True
    links.new(rim.outputs["Value"], wisp.inputs[0])
    links.new(stroke.outputs["Fac"], wisp.inputs[1])
    wisp_gain = nodes.new("ShaderNodeMath")
    wisp_gain.operation = "MULTIPLY"
    wisp_gain.inputs[1].default_value = 0.55
    links.new(wisp.outputs["Value"], wisp_gain.inputs[0])

    mask = nodes.new("ShaderNodeMath")
    mask.operation = "ADD"
    mask.use_clamp = True
    links.new(radial.outputs["Result"], mask.inputs[0])
    links.new(wisp_gain.outputs["Value"], mask.inputs[1])
    return mask.outputs["Value"], stroke.outputs["Fac"]


def _apply_patch_overlay(mat: bpy.types.Material, seed: int) -> bpy.types.Material:
    """Shift copied wall paint: slightly different HSV + glossier, masked to the blob."""
    if not mat.use_nodes or mat.node_tree is None:
        return mat
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    rng = np.random.default_rng(seed)

    principled = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        return mat

    mask_sock, stroke_sock = _build_patch_mask(nt, seed)

    darker = rng.random() < 0.8
    hue = 0.5 + float(rng.uniform(-0.02, 0.02))
    sat = float(rng.uniform(1.18, 1.42))
    val = float(rng.uniform(0.70, 0.86) if darker else rng.uniform(1.08, 1.18))
    gloss = float(rng.uniform(0.50, 0.70))

    base = principled.inputs.get("Base Color")
    if base is not None:
        hsv = nodes.new("ShaderNodeHueSaturation")
        hsv.inputs["Hue"].default_value = hue
        hsv.inputs["Saturation"].default_value = sat
        hsv.inputs["Value"].default_value = val
        hsv.inputs["Fac"].default_value = 1.0
        incoming = [lk for lk in links if lk.to_socket == base]
        if incoming:
            src = incoming[0].from_socket
            links.remove(incoming[0])
            links.new(src, hsv.inputs["Color"])
        else:
            hsv.inputs["Color"].default_value = tuple(base.default_value)
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MIX"
        links.new(mask_sock, mix.inputs["Fac"])
        links.new(hsv.outputs["Color"], mix.inputs["Color2"])
        if incoming:
            links.new(src, mix.inputs["Color1"])
        else:
            mix.inputs["Color1"].default_value = tuple(base.default_value)
        links.new(mix.outputs["Color"], base)

    rough = principled.inputs.get("Roughness")
    if rough is not None:
        # roughness' = roughness * (1 - mask * (1 - gloss))
        r_in = [lk for lk in links if lk.to_socket == rough]
        scale = nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = 1.0 - gloss
        links.new(mask_sock, scale.inputs[0])
        keep = nodes.new("ShaderNodeMath")
        keep.operation = "SUBTRACT"
        keep.inputs[0].default_value = 1.0
        links.new(scale.outputs["Value"], keep.inputs[1])
        mul = nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        if r_in:
            src = r_in[0].from_socket
            links.remove(r_in[0])
            links.new(src, mul.inputs[0])
        else:
            mul.inputs[0].default_value = float(rough.default_value)
        links.new(keep.outputs["Value"], mul.inputs[1])
        links.new(mul.outputs["Value"], rough)

    alpha = principled.inputs.get("Alpha")
    if alpha is not None:
        for lk in list(links):
            if lk.to_socket == alpha:
                links.remove(lk)
        links.new(mask_sock, alpha)

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = float(rng.uniform(0.08, 0.18))
    bump.inputs["Distance"].default_value = 0.0012
    hmul = nodes.new("ShaderNodeMath")
    hmul.operation = "MULTIPLY"
    links.new(stroke_sock, hmul.inputs[0])
    links.new(mask_sock, hmul.inputs[1])
    links.new(hmul.outputs["Value"], bump.inputs["Height"])
    nrm = principled.inputs.get("Normal")
    if nrm is not None:
        n_in = [lk for lk in links if lk.to_socket == nrm]
        if n_in:
            src = n_in[0].from_socket
            links.remove(n_in[0])
            links.new(src, bump.inputs["Normal"])
        links.new(bump.outputs["Normal"], nrm)

    _set_blend(mat)
    return mat


def create_paint_patch_material(name: str, seed: int) -> bpy.types.Material:
    """Fallback plaster-like patch when the room wall material is not available yet."""
    with FixedSeed(seed):
        v = float(np.random.uniform(0.72, 0.90))
        color = (v, v * 0.99, v * 0.96, 1.0)
        rough = float(np.random.uniform(0.58, 0.78))

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.22
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    _apply_patch_overlay(mat, seed)
    return mat


class PaintPatchPlaneFactory(AssetFactory):
    """Uneven finish / touch-up mismatch: a feathered blob almost the wall color."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.plane_size = np.random.uniform(0.38, 0.70)

    def create_placeholder(self, **kwargs):
        ph = new_bbox(-0.005, 0.005, -0.5, 0.5, -0.5, 0.5)
        butil.modify_mesh(ph, "TRIANGULATE", min_vertices=3)
        tag_canonical_surfaces(ph)
        return ph

    def create_asset(self, placeholder=None, **kwargs) -> bpy.types.Object:
        plane = new_plane()
        geom_seed = int_hash((self.factory_seed, kwargs.get("i", 0), "geom"))
        with FixedSeed(geom_seed):
            scale_z_val = np.random.uniform(0.7, 1.0) * self.plane_size / 2
            scale_y_val = np.random.uniform(0.7, 1.0) * self.plane_size / 2
        plane.scale = (scale_z_val, scale_y_val, 1.0)
        plane.rotation_euler = (0.0, np.pi / 2, 0.0)
        butil.apply_transform(plane, loc=False, rot=True, scale=True)
        mat = create_paint_patch_material(
            name=f"PaintPatchMaterial_{geom_seed}",
            seed=int_hash((self.factory_seed, kwargs.get("i", 0))),
        )
        plane.data.materials.append(mat)
        plane.name = f"PaintPatchPlane_{geom_seed}"
        plane.visible_shadow = False
        plane.visible_diffuse = True
        return plane

    def finalize_assets(
        self, assets, state=None, wall_by_name=None, update_embed_transform=True
    ):
        # Sit a millimetre off the wall into the room so the decal is not buried.
        EMBED_INTO_ROOM = 0.0018

        from infinigen.core import tags as t

        if wall_by_name is None:
            wall_by_name = {
                w.name: w
                for w in bpy.data.objects
                if w.name.endswith(".wall") and w.type == "MESH"
            }

        def _get_wall_mat(patch_obj):
            if state is not None and wall_by_name:
                for os in state.objs.values():
                    if os.obj is not patch_obj:
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
            nearest_wall, best_d2 = None, None
            center = patch_obj.matrix_world.translation
            for w in wall_by_name.values():
                d2 = (center - w.matrix_world.translation).length_squared
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    nearest_wall = w
            if nearest_wall is not None:
                for mat in nearest_wall.data.materials:
                    if mat is not None:
                        return mat
            for w in wall_by_name.values():
                for mat in w.data.materials:
                    if mat is not None:
                        return mat
            return None

        for obj in assets:
            if obj.type != "MESH" or not obj.data.polygons:
                continue
            try:
                wall_mat = _get_wall_mat(obj)
                if wall_mat is not None:
                    seed = int_hash(obj.name)
                    patch_mat = _apply_patch_overlay(wall_mat.copy(), seed)
                    patch_mat.name = f"PaintPatchMaterial_{seed}"
                    if obj.data.materials:
                        obj.data.materials[0] = patch_mat
                    else:
                        obj.data.materials.append(patch_mat)
                if update_embed_transform:
                    into_room = obj.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))
                    if into_room.length > 1e-8:
                        obj.location += into_room.normalized() * EMBED_INTO_ROOM
            except Exception as e:
                logger.warning("Failed to embed paint-patch plane %s: %s", obj.name, e)


def refresh_paint_patch_materials(wall_objects):
    patches = [
        o
        for o in bpy.data.objects
        if o.type == "MESH"
        and (
            o.name.startswith("PaintPatchPlane")
            or "PaintPatchPlaneFactory" in o.name
        )
    ]
    if not patches:
        return
    wall_by_name = {w.name: w for w in wall_objects if w and w.type == "MESH"}
    if not wall_by_name:
        return
    PaintPatchPlaneFactory(factory_seed=0).finalize_assets(
        patches, state=None, wall_by_name=wall_by_name, update_embed_transform=False
    )
