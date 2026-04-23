# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import logging

import bpy

from infinigen.assets.lighting import sky_lighting
from infinigen.assets.objects import rocks, trees
from infinigen.assets.wall_bubble_plane import refresh_wall_bubble_materials
from infinigen.core.nodes.node_utils import resample_node_group
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.util.logging import Timer
from infinigen.core.util.math import FixedSeed, int_hash

logger = logging.getLogger(__name__)


def resample_all(factory_class):
    if "placeholders" not in bpy.data.collections:
        return
    for placeholder_col in bpy.data.collections["placeholders"].children:
        if "(" not in placeholder_col.name:
            continue
        classname, _ = placeholder_col.name.split("(", 1)
        if classname != factory_class.__name__:
            continue

        placeholders = [o for o in placeholder_col.objects if o.parent is None]
        for pholder in placeholders:
            factory_class.quickly_resample(pholder)


def resample_room_surfaces(scene_seed):
    """Re-apply wall, floor, and ceiling materials using a new seed.

    This lets the render step produce different material appearances for
    the same apartment geometry.  The collections written by split_rooms
    during coarse generation are used to locate the meshes.
    """
    from infinigen.core.constraints.example_solver.room import decorate as room_dec
    from infinigen.core.constraints.constraint_language.constants import RoomConstants

    constants = RoomConstants()

    wall_col = bpy.data.collections.get("unique_assets:room_wall")
    floor_col = bpy.data.collections.get("unique_assets:room_floor")
    ceiling_col = bpy.data.collections.get("unique_assets:room_ceiling")

    if wall_col and wall_col.objects:
        with Timer("Resample wall materials"):
            room_dec.room_walls(
                list(wall_col.objects),
                constants,
                material_seed=int_hash((scene_seed, "render_walls")),
            )
        with Timer("Refresh wall-bubble materials"):
            refresh_wall_bubble_materials(list(wall_col.objects))
    if floor_col and floor_col.objects:
        with Timer("Resample floor materials"):
            room_dec.room_floors(
                list(floor_col.objects),
                material_seed=int_hash((scene_seed, "render_floors")),
            )
    if ceiling_col and ceiling_col.objects:
        with Timer("Resample ceiling materials"):
            room_dec.room_ceilings(
                list(ceiling_col.objects),
                material_seed=int_hash((scene_seed, "render_ceilings")),
            )


def resample_scene(scene_seed):
    with FixedSeed(scene_seed), Timer("Resample noise nodes in materials"):
        for material in bpy.data.materials:
            nw = NodeWrangler(material.node_tree)
            resample_node_group(nw, scene_seed)

    with FixedSeed(scene_seed), Timer("Resample noise nodes in scatters"):
        for obj in bpy.data.objects:
            for modifier in obj.modifiers:
                if not any(
                    obj.name.startswith(s)
                    for s in ["BlenderRockFactory", "CloudFactory"]
                ):
                    if modifier.type == "NODES":
                        nw = NodeWrangler(modifier.node_group)
                        resample_node_group(nw, scene_seed)

    with (
        FixedSeed(scene_seed),
        Timer("Resample all placeholders"),
    ):  # CloudFactory too expensive
        resample_all(rocks.GlowingRocksFactory)
        resample_all(trees.TreeFactory)
        resample_all(trees.BushFactory)
        # resample_all(CreatureFactory)
    with FixedSeed(scene_seed):
        sky_lighting.add_lighting()

    resample_room_surfaces(scene_seed)
