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

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util.math import int_hash

logger = logging.getLogger(__name__)

BEVEL_MODIFIER_NAME = "SynDefectEdgeBevel"
IMPERFECTION_GROUP_NAME = "SynDefectSurfaceImperfection"

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
