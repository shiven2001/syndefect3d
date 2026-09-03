# Copyright (C) 2026, SynDefect3D.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""Render-time micro-detail: edge bevels and surface-imperfection roughness.

Two of the loudest CGI tells in a procedural interior are geometric and
shading, not tonal:

* every edge is a mathematically sharp arris, so no corner catches the
  highlight a real painted reveal or door casing does;
* every surface has one uniform roughness, so reflections and sheen are flat
  across a wall that in life carries smudges, dust and hand grease.

Both passes below default to no-op and are switched on from ``realism_v2.gin``.
They run from ``apply_realism_adjustments`` at render time, so they also apply
to blends coarsed before this existed.
"""

import logging
import math

import bpy
import gin
import numpy as np

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util.math import int_hash

logger = logging.getLogger(__name__)

BEVEL_MODIFIER_NAME = "SynDefectEdgeBevel"
IMPERFECTION_GROUP_NAME = "SynDefectSurfaceImperfection"
MICRONORMAL_GROUP_NAME = "SynDefectMicroNormal"

# Materials that must never receive a bump chain: a normal perturbation on
# glass frosts it, on a mirror it destroys the reflection, and on an emitter
# it does nothing but cost samples.
NO_BUMP_TOKENS = (
    "glass", "mirror", "emission", "light", "lamp", "screen", "display",
    "water", "chrome",
)

# Mirrors the prefix -> class map in tools/prepare_defect_annotated_dataset.py.
# Defect planes carry the annotation signal: their geometry must not move and
# their shaders must not be perturbed, or the material-index masks stop lining
# up with the RGB they are exported against.
DEFECT_MATERIAL_PREFIXES = (
    "CrackMaterial",
    "PaintPeelMaterial",
    "SpallingMaterial",
    "SpallingPlugMaterial",
    "BubbleMaterial",
    "OpenWiringMaterial",
    "PaintRunMaterial",
    "PaintPatchMaterial",
    "CornerChipMaterial",
    "TileChipMaterial",
)


def is_defect_material(mat) -> bool:
    return mat is not None and (mat.name or "").startswith(DEFECT_MATERIAL_PREFIXES)


def _has_defect_material(obj) -> bool:
    return any(is_defect_material(m) for m in obj.data.materials)


@gin.configurable
def add_edge_bevels(
    enabled=False,
    width=0.002,
    segments=3,
    angle_deg=30.0,
    harden_normals=False,
    max_polygons=100_000,
    skip_defects=True,
):
    """Add a small non-destructive edge bevel to every mesh in the scene.

    A modifier rather than a bmesh op: reversible, absent from the saved
    topology, and re-runnable. ``max_polygons`` skips dense assets where a
    2 mm chamfer is sub-pixel anyway but the added geometry is not free.
    """
    if not enabled:
        return

    angle = math.radians(float(angle_deg))
    n_added = n_skipped = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        # Idempotent: the render-time hooks can run more than once per session.
        if obj.modifiers.get(BEVEL_MODIFIER_NAME) is not None:
            continue
        n_poly = len(obj.data.polygons)
        if n_poly == 0:
            continue
        if max_polygons is not None and n_poly > int(max_polygons):
            n_skipped += 1
            continue
        if skip_defects and _has_defect_material(obj):
            n_skipped += 1
            continue

        mod = obj.modifiers.new(BEVEL_MODIFIER_NAME, "BEVEL")
        mod.width = float(width)
        mod.segments = int(segments)
        mod.limit_method = "ANGLE"
        mod.angle_limit = angle
        mod.miter_outer = "MITER_ARC"
        mod.use_clamp_overlap = True
        # Harden normals needs smooth shading; on a flat-shaded mesh Blender
        # only warns, so leave it opt-in rather than guessing per object.
        mod.harden_normals = bool(harden_normals)
        n_added += 1

    logger.info("edge bevel: beveled %s meshes, skipped %s", n_added, n_skipped)


@node_utils.to_nodegroup(
    IMPERFECTION_GROUP_NAME, singleton=True, type="ShaderNodeTree"
)
def nodegroup_surface_imperfection(nw: NodeWrangler):
    """Roughness break-up: broad smudges over fine dust/grease grain.

    Object coordinates, so the pattern is fixed to the asset and does not swim
    between the main camera and its defect close-up rig.
    """
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Roughness", 0.5000),
            ("NodeSocketFloat", "Scale", 6.0000),
            ("NodeSocketFloat", "Strength", 0.0600),
            ("NodeSocketFloat", "Seed", 0.0000),
        ],
    )

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    scale = group_input.outputs["Scale"]

    # Broad, low-frequency wipe marks and hand grease.
    smudge = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": scale,
            "Detail": 6.0000,
            "Roughness": 0.5500,
            "Distortion": 0.3000,
        },
        attrs={"noise_dimensions": "4D"},
    )

    # Fine settled dust / fingerprint grain. Musgrave is routed through the
    # 4.x compatibility shim (Blender 4.1 folded it into Noise Texture).
    dust = nw.new_node(
        Nodes.MusgraveTexture,
        input_kwargs={
            "Vector": texture_coordinate.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": nw.scalar_multiply(scale, 18.0000),
            "Detail": 8.0000,
            "Dimension": 1.4000,
            "Lacunarity": 2.2000,
        },
        attrs={"musgrave_dimensions": "4D"},
    )

    blend = nw.scalar_add(
        nw.scalar_multiply(smudge.outputs["Fac"], 0.6000),
        nw.scalar_multiply(dust.outputs["Fac"], 0.4000),
    )

    # Recentre on zero so the pass perturbs roughness both ways instead of
    # only ever roughening (which would dull every surface in the scene).
    centered = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": blend, 1: 0.2500, 2: 0.7500, 3: -1.0000, 4: 1.0000},
    )

    perturbed = nw.scalar_add(
        group_input.outputs["Roughness"],
        nw.scalar_multiply(centered.outputs["Result"], group_input.outputs["Strength"]),
    )

    clamped = nw.new_node(
        Nodes.Clamp,
        input_kwargs={"Value": perturbed, "Min": 0.0300, "Max": 1.0000},
    )

    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Roughness": clamped},
        attrs={"is_active_output": True},
    )


def _is_transmissive(node, max_transmission: float) -> bool:
    sock = node.inputs.get("Transmission Weight") or node.inputs.get("Transmission")
    if sock is None:
        return False
    if sock.links:
        return True
    return float(sock.default_value) > max_transmission


def _insert_imperfection(material, node, group, *, strength, scale, seed):
    sock = node.inputs.get("Roughness")
    if sock is None:
        return False

    nw = NodeWrangler(material.node_tree)
    if sock.links:
        src = sock.links[0].from_socket
        if getattr(src.node, "node_tree", None) is group:
            return False  # already wired by an earlier run
        roughness_in = src
    else:
        roughness_in = float(sock.default_value)

    group_node = nw.new_node(
        group.name,
        input_kwargs={
            "Roughness": roughness_in,
            "Scale": float(scale),
            "Strength": float(strength),
            "Seed": float(seed),
        },
    )
    material.node_tree.links.new(group_node.outputs["Roughness"], sock)
    return True


@node_utils.to_nodegroup(MICRONORMAL_GROUP_NAME, singleton=True, type="ShaderNodeTree")
def nodegroup_micro_normal(nw: NodeWrangler):
    """Fine surface relief for assets that ship with a flat Normal input.

    An audit of a solved in-use scene found 274 of 380 furniture materials -
    98% of furniture surface area - with nothing driving Normal at all. Flat
    albedo under flat shading is the loudest CGI tell there is, and it is the
    same failure the defect assets had: everything painted, nothing in relief.

    Two octaves in OBJECT coordinates so the grain is fixed to the asset and
    does not swim between the room camera and its close-up rig: a broad weave
    or grain direction, plus fine tooth on top.
    """
    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Scale", 220.0000),
            ("NodeSocketFloat", "Strength", 0.1500),
            ("NodeSocketFloat", "Distance", 0.0006),
            ("NodeSocketFloat", "Seed", 0.0000),
        ],
    )
    tex = nw.new_node(Nodes.TextureCoord)
    scale = group_input.outputs["Scale"]

    coarse = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": tex.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": scale,
            "Detail": 6.0000,
            "Roughness": 0.6000,
        },
        attrs={"noise_dimensions": "4D"},
    )
    fine = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Vector": tex.outputs["Object"],
            "W": group_input.outputs["Seed"],
            "Scale": nw.scalar_multiply(scale, 5.0000),
            "Detail": 8.0000,
            "Roughness": 0.7500,
        },
        attrs={"noise_dimensions": "4D"},
    )
    height = nw.scalar_add(
        nw.scalar_multiply(coarse.outputs["Fac"], 0.6500),
        nw.scalar_multiply(fine.outputs["Fac"], 0.3500),
    )
    # strict=False: infinigen bans Bump in favour of true Displacement, but
    # true displacement needs adaptive subdivision (not enabled in this repo -
    # displacement_method="BOTH" measurably does nothing here) and would be
    # ruinous across every furniture surface. Sub-millimetre grain is exactly
    # what bump is for.
    bump = nw.new_node(
        Nodes.Bump,
        input_kwargs={
            "Strength": group_input.outputs["Strength"],
            "Distance": group_input.outputs["Distance"],
            "Height": height,
        },
        strict=False,
    )
    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Normal": bump.outputs["Normal"]},
        attrs={"is_active_output": True},
    )


def _skip_for_bump(mat, node, max_transmission: float) -> bool:
    name = (mat.name or "").lower()
    if any(tok in name for tok in NO_BUMP_TOKENS):
        return True
    if _is_transmissive(node, max_transmission):
        return True
    for n in mat.node_tree.nodes:
        if n.bl_idname in (Nodes.GlassBSDF, Nodes.Emission, Nodes.TranslucentBSDF):
            return True
    return False


def _desaturate(mat, node, max_sat: float) -> bool:
    """Pull an over-saturated base colour back toward plausible timber/fabric.

    The audit found 54 materials at saturation > 0.5 over 31% of furniture
    area, wood shaders as high as 0.79; real timber sits nearer 0.25-0.45.

    A procedural colour cannot be evaluated at build time, so the Hue/Saturation
    node's multiplier is derived from the material's *measured* mean saturation
    (`_mean_material_rgb` over the whole tree): mult = max_sat / measured. A
    fixed multiplier would barely touch a 0.79 shader while over-flattening a
    0.5 one. An unlinked colour is clamped in place instead.
    """
    import colorsys

    # Local import: core/ should not depend on assets/ at module scope.
    from infinigen.assets.crack_plane import _mean_material_rgb

    sock = node.inputs.get("Base Color")
    if sock is None:
        return False

    if not sock.links:
        r, g, b, a = sock.default_value
        h, sv, v = colorsys.rgb_to_hsv(*[min(max(c, 0.0), 1.0) for c in (r, g, b)])
        if sv <= max_sat:
            return False
        r2, g2, b2 = colorsys.hsv_to_rgb(h, max_sat, v)
        sock.default_value = (r2, g2, b2, a)
        return True

    src = sock.links[0].from_socket
    if src.node.bl_idname == "ShaderNodeHueSaturation":
        return False  # already spliced by an earlier run

    rgb = _mean_material_rgb(mat)
    if rgb is None:
        return False
    _h, measured, _v = colorsys.rgb_to_hsv(*[min(max(c, 0.0), 1.0) for c in rgb])
    if measured <= max_sat:
        return False
    mult = float(np.clip(max_sat / max(measured, 1e-6), 0.05, 1.0))

    nw = NodeWrangler(mat.node_tree)
    hsv = nw.new_node(
        "ShaderNodeHueSaturation",
        input_kwargs={"Saturation": mult, "Color": src},
    )
    mat.node_tree.links.new(hsv.outputs["Color"], sock)
    return True


def _insert_micro_normal(mat, node, group, *, strength, scale, distance, seed) -> bool:
    sock = node.inputs.get("Normal")
    if sock is None or sock.links:
        return False  # already has a normal map / bump; leave the author's work
    nw = NodeWrangler(mat.node_tree)
    gn = nw.new_node(
        group.name,
        input_kwargs={
            "Scale": float(scale),
            "Strength": float(strength),
            "Distance": float(distance),
            "Seed": float(seed),
        },
    )
    mat.node_tree.links.new(gn.outputs["Normal"], sock)
    return True


@gin.configurable
def apply_furniture_material_realism(
    enabled=False,
    micro_normal=True,
    desaturate=True,
    max_saturation=0.45,
    bump_strength=0.15,
    bump_scale=220.0,
    bump_distance=0.0006,
    seed=0.0,
    skip_defects=True,
    max_transmission=0.05,
):
    """Give furniture shaders surface relief and sane saturation.

    Walls, ceilings and floors are handled by their own shaders; this targets
    what the in-use configs add on top - beds, cabinets, shelving, upholstery -
    where the audit showed 98% of surface area with no Normal input and a third
    of it over-saturated. Runs at render time like the other passes here, so it
    also applies to blends solved before it existed, and is a no-op by default.

    Defect materials are skipped: they carry the annotation signal.
    """
    if not enabled:
        return

    group = nodegroup_micro_normal() if micro_normal else None
    n_norm = n_sat = n_skip = 0
    for mat in bpy.data.materials:
        if mat is None or mat.node_tree is None or mat.users == 0:
            continue
        if skip_defects and is_defect_material(mat):
            continue
        for node in list(mat.node_tree.nodes):
            if node.bl_idname != Nodes.PrincipledBSDF:
                continue
            if _skip_for_bump(mat, node, float(max_transmission)):
                n_skip += 1
                continue
            if desaturate:
                n_sat += int(_desaturate(mat, node, float(max_saturation)))
            if micro_normal:
                mat_seed = float(seed) + (int_hash(mat.name) % 997) * 0.1
                n_norm += int(
                    _insert_micro_normal(
                        mat, node, group,
                        strength=bump_strength, scale=bump_scale,
                        distance=bump_distance, seed=mat_seed,
                    )
                )

    logger.info(
        "furniture realism: micro-normal on %s materials, desaturated %s, skipped %s",
        n_norm, n_sat, n_skip,
    )


@gin.configurable
def apply_surface_imperfections(
    enabled=False,
    strength=0.0600,
    scale=6.0000,
    seed=0.0000,
    skip_defects=True,
    skip_transmissive=True,
    max_transmission=0.0500,
):
    """Wire the imperfection group into the Roughness of every scene material.

    Skips defect shaders (they are the label) and glass (perturbing roughness
    there frosts the windows the daylight comes through).
    """
    if not enabled or strength is None or abs(float(strength)) < 1e-6:
        return

    group = nodegroup_surface_imperfection()
    n_mats = 0
    base_seed = float(seed)
    for mat in bpy.data.materials:
        if mat is None or mat.node_tree is None or mat.users == 0:
            continue
        if skip_defects and is_defect_material(mat):
            continue
        touched = False
        for node in list(mat.node_tree.nodes):
            if node.bl_idname != Nodes.PrincipledBSDF:
                continue
            if skip_transmissive and _is_transmissive(node, float(max_transmission)):
                continue
            # Object coordinates repeat across identically-built assets, so
            # offset the seed per material or every wall shares one smudge.
            mat_seed = base_seed + (int_hash(mat.name) % 997) * 0.1000
            touched |= _insert_imperfection(
                mat, node, group, strength=strength, scale=scale, seed=mat_seed
            )
        n_mats += int(touched)

    logger.info("surface imperfection: patched roughness on %s materials", n_mats)


def disable_crack_plane_lighting():
    """Stop crack decals shading the wall/ceiling/floor behind them.

    Cycles still traces shadow and GI rays through unused film even when
    the shader is Transparent. A hairline crack should not cast a card-shaped
    contact shadow; only the camera should see the fissure.
    """
    n_obj = n_mat = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        mats = [m for m in obj.data.materials if m is not None]
        if not any(
            (m.name or "").startswith(
                ("CrackMaterial", "CornerChipMaterial", "TileChipMaterial")
            )
            for m in mats
        ):
            continue
        obj.visible_shadow = False
        obj.visible_diffuse = False
        if hasattr(obj, "visible_glossy"):
            obj.visible_glossy = False
        if hasattr(obj, "visible_transmission"):
            obj.visible_transmission = False
        if hasattr(obj, "visible_volume_scatter"):
            obj.visible_volume_scatter = False
        n_obj += 1
        for mat in mats:
            if not (mat.name or "").startswith(
                ("CrackMaterial", "CornerChipMaterial", "TileChipMaterial")
            ):
                continue
            if hasattr(mat, "use_transparent_shadow"):
                mat.use_transparent_shadow = False
            if hasattr(mat, "shadow_method"):
                mat.shadow_method = "NONE"
            n_mat += 1
    if n_obj:
        logger.info(
            "Disabled lighting rays on %s crack/chip planes (%s materials)", n_obj, n_mat
        )
