# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: populate_state_placeholders, apply_cutter
# - Stamatis Alexandropoulos: Initial version of window cutting

import logging

import bpy
from tqdm import tqdm

from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import usage_lookup
from infinigen.core.constraints.constraint_language.util import delete_obj
from infinigen.core.constraints.example_solver.geometry import parse_scene
from infinigen.core.constraints.example_solver.state_def import State
from infinigen.core.placement.placement import parse_asset_name
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)

# Filled by populate_state_placeholders; consumed by generate_indoors after room_walls
deferred_wall_bubble_finalize = []


def apply_cutter(state, objkey, cutter):
    os = state.objs[objkey]

    cut_objs = []
    for i, relation_state in enumerate(os.relations):
        # TODO in theory we maybe should check if they actually intersect

        parent_obj = state.objs[relation_state.target_name].obj
        butil.modify_mesh(
            parent_obj,
            "BOOLEAN",
            object=butil.copy(cutter),
            operation="DIFFERENCE",
            solver="FAST",
        )

        target_obj_name = state.objs[relation_state.target_name].obj.name
        cut_objs.append((relation_state.target_name, target_obj_name))

    cutter_col = butil.get_collection("placeholders:asset_cutters")
    butil.put_in_collection(cutter, cutter_col)

    return cut_objs


def populate_state_placeholders(state: State, filter=None, final=True):
    logger.info(f"Populating placeholders {final=} {filter=}")
    unique_assets = butil.get_collection("unique_assets")
    unique_assets.hide_viewport = True

    if final:
        for os in state.objs.values():
            if t.Semantics.Room in os.tags:
                os.obj = bpy.data.objects[os.obj.name + ".meshed"]

    targets = []

    for objkey, os in state.objs.items():
        if os.generator is None:
            continue

        if filter is not None and not usage_lookup.has_usage(
            os.generator.__class__, filter
        ):
            continue

        if "spawn_asset" in os.obj.name:
            butil.put_in_collection(os.obj, unique_assets)
            logger.debug(f"Found already populated asset {os.obj.name=}, continuing")
            continue

        targets.append(objkey)

    update_state_mesh_objs = []
    deferred_spalling_plug_objs = []
    deferred_wall_bubble_objs = []

    for i, objkey in enumerate(targets):
        os = state.objs[objkey]
        placeholder = os.obj

        logger.info(f"Populating {i}/{len(targets)} {placeholder.name=}")

        old_objname = placeholder.name
        update_state_mesh_objs.append((objkey, old_objname))

        *_, inst_seed = parse_asset_name(placeholder.name)
        os.obj = os.generator.spawn_asset(
            i=int(inst_seed),
            loc=placeholder.location,  # we could use placeholder=pholder here, but I worry pholder may have been modified
            rot=placeholder.rotation_euler,
        )
        # Defer SpallingPlugPlaneFactory finalize_assets until after all assets are populated,
        # so wall plugs exist when we snap spalling plugs to them.
        if os.generator.__class__.__name__ == "SpallingPlugPlaneFactory":
            deferred_spalling_plug_objs.append((os.generator, os.obj))
        # Defer WallBubblePlaneFactory finalize_assets until after split_rooms + room_walls,
        # so .wall objects with materials exist.
        elif os.generator.__class__.__name__ == "WallBubblePlaneFactory":
            deferred_wall_bubble_objs.append((os.generator, os.obj))
        else:
            os.generator.finalize_assets([os.obj])
        butil.put_in_collection(os.obj, unique_assets)

        cutter = next(
            (o for o in butil.iter_object_tree(os.obj) if o.name.endswith(".cutter")),
            None,
        )
        logger.debug(
            f"{populate_state_placeholders.__name__} found {cutter=} for {os.obj.name=}"
        )
        if cutter is not None:
            cut_objs = apply_cutter(state, objkey, cutter)
            logger.debug(
                f"{populate_state_placeholders.__name__} cut {cutter.name=} from {cut_objs=}"
            )
            update_state_mesh_objs += cut_objs

    # Run deferred SpallingPlugPlaneFactory finalize_assets now that all wall plugs exist
    for generator, obj in deferred_spalling_plug_objs:
        logger.info(f"Finalizing deferred spalling plug: {obj.name}")
        generator.finalize_assets([obj])

    # WallBubblePlaneFactory finalize runs after room_walls (see generate_indoors)
    deferred_wall_bubble_finalize.clear()
    deferred_wall_bubble_finalize.extend(deferred_wall_bubble_objs)

    unique_assets.hide_viewport = False

    if final:
        return

    # objects modified in any way (via pholder update or boolean cut) must be synched with trimesh state
    for objkey, old_objname in tqdm(
        set(update_state_mesh_objs), desc="Updating trimesh with populated objects"
    ):
        os = state.objs[objkey]

        # delete old trimesh
        delete_obj(state.trimesh_scene, old_objname, delete_blender=False)

        # put the new, populated object into the state
        parse_scene.preprocess_obj(os.obj)
        if not final:
            tagging.tag_canonical_surfaces(os.obj)
        parse_scene.add_to_scene(state.trimesh_scene, os.obj, preprocess=True)
