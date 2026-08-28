# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

import importlib
import logging
import os
from collections import defaultdict

import bmesh
import bpy
import gin
import numpy as np
import shapely
import shapely.affinity
import trimesh.convex
from numpy.random import uniform
from shapely import Point
from shapely.ops import nearest_points
from tqdm import tqdm, trange
from trimesh.transformations import translation_matrix

import infinigen.core.surface as surface
from infinigen.assets.composition import material_assignments
from infinigen.assets.materials.ceramic import plaster
from infinigen.assets.objects.elements import PillarFactory, random_staircase_factory
from infinigen.assets.objects.elements.doors import random_door_factory
from infinigen.assets.objects.windows import WindowFactory
from infinigen.assets.objects.windows.window_handle import make_window_handle
from infinigen.assets.utils.decorate import (
    read_center,
    read_co,
    read_edge_length,
    read_edges,
    remove_vertices,
)
from infinigen.assets.utils.object import join_objects, obj2trimesh
from infinigen.assets.utils.shapes import dissolve_limited, obj2polygon
from infinigen.assets.utils.uv import unwrap_normal
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.constraint_language.constants import RoomConstants
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.room.base import room_level, room_type
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.random import log_uniform, weighted_sample
from infinigen.core.util.random import random_general as rg

logger = logging.getLogger(__name__)


def _placeholder_interior_y_sign(placeholder, state):
    """+1 if placeholder +Y points into room volume, else -1 (keep doors inward)."""
    if state is None or placeholder is None:
        return 1.0
    from mathutils import Vector

    y_axis = placeholder.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    origin = placeholder.matrix_world.translation
    toward = 0.0
    away = 0.0
    for os in getattr(state, "objs", {}).values():
        tags = getattr(os, "tags", set()) or set()
        if t.Semantics.Room not in tags:
            continue
        obj = getattr(os, "obj", None)
        if obj is None or obj.type != "MESH" or not getattr(obj, "bound_box", None):
            continue
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        center = sum(corners, Vector((0.0, 0.0, 0.0))) / max(len(corners), 1)
        score = float((center - origin).dot(y_axis))
        if score > 0.05:
            toward += score
        elif score < -0.05:
            away += -score
    if away > toward:
        return -1.0
    return 1.0


def _door_cutout_local_x(factory, door_width):
    """Parent-space X so the leaf fills the cutter, not a randomized offset.

    ``populate_doors`` always used ``+door_width/2``, which only matches a
    right-hinged ``single_column`` door. ``full_frame_*`` shifts the leaf by
    half a width; left-hinge flips it the other way.
    """
    style = getattr(factory, "door_frame_style", "single_column")
    orient = getattr(factory, "door_orientation", "right")
    if style in {
        "full_frame_square",
        "full_frame_dome",
        "full_frame_double_door",
    }:
        return 0.0
    if orient == "left":
        return -door_width / 2
    return door_width / 2


def split_rooms(rooms_meshed: list[bpy.types.Object]):
    extract_tags = {
        "wall": {t.Subpart.Wall, t.Subpart.Visible},
        "floor": {t.Subpart.SupportSurface, t.Subpart.Visible},
        "ceiling": {t.Subpart.Ceiling, t.Subpart.Visible},
    }

    meshes = {
        n: [tagging.extract_tagged_faces(r, tags) for r in rooms_meshed]
        for n, tags in extract_tags.items()
    }

    for k, ms in meshes.items():
        m2delete = []
        for m in ms:
            if m.name.startswith("vert"):
                butil.select_none()
                butil.delete(m)
                m2delete.append(m)
        for m in m2delete:
            ms.remove(m)

    meshes["exterior"] = [
        tagging.extract_mask(r, 1 - tagging.tagged_face_mask(r, t.Subpart.Visible))
        for r in rooms_meshed
    ]

    for n, objs in meshes.items():
        for o in objs:
            o.name = o.name.split(".")[0] + f".{n}"
        butil.origin_set(objs, "ORIGIN_GEOMETRY", center="MEDIAN")

    meshes = {
        n: butil.put_in_collection(objs, "unique_assets:room_" + n)
        for n, objs in meshes.items()
    }

    return meshes


def import_material(factory_name):
    with gin.unlock_config(), FixedSeed(0):
        try:
            return importlib.import_module(f"infinigen.assets.materials.{factory_name}")
        except ImportError:
            for subdir in os.listdir("infinigen/assets/materials"):
                if not subdir.endswith(".py"):
                    with gin.unlock_config():
                        module = importlib.import_module(
                            f"infinigen.assets.materials.{subdir.split('.')[0]}"
                        )
                    if hasattr(module, factory_name):
                        return getattr(module, factory_name)
            else:
                raise Exception(f"{factory_name} not Found.")


room_ceiling_fns = defaultdict(
    lambda: material_assignments.ceiling,
    {
        t.Semantics.Warehouse: material_assignments.warehouse_ceiling,
        t.Semantics.Garage: material_assignments.garage_ceiling,
    },
)
# TODO: add wall art and mirror to the walls
room_floor_fns = defaultdict(
    lambda: material_assignments.floor,
    {
        t.Semantics.Garage: material_assignments.garage_floor,
        t.Semantics.Utility: material_assignments.utility_floor,
        t.Semantics.Bathroom: material_assignments.bathroom_floor,
        t.Semantics.Restroom: material_assignments.bathroom_floor,
        t.Semantics.Balcony: material_assignments.balcony_floor,
        t.Semantics.Office: material_assignments.office_floor,
        t.Semantics.FactoryOffice: material_assignments.office_floor,
        t.Semantics.OpenOffice: material_assignments.office_floor,
        t.Semantics.Warehouse: material_assignments.warehouse_floor,
    },
)
# TODO: add wall art and mirror to the walls
room_wall_fns = defaultdict(
    lambda: material_assignments.wall,
    {
        t.Semantics.Kitchen: material_assignments.kitchen_wall,
        t.Semantics.Garage: material_assignments.garage_wall,
        t.Semantics.Utility: material_assignments.utility_wall,
        t.Semantics.Balcony: material_assignments.balcony_wall,
        t.Semantics.Bathroom: material_assignments.bathroom_wall,
        t.Semantics.Restroom: material_assignments.bathroom_wall,
        t.Semantics.Warehouse: material_assignments.warehouse_wall,
    },
)

# noinspection PyTypeChecker
# abstract_art (Art/DarkArt) removed - uses text_texture; walls use plaster only
# Use "none" only so all sides of wall have same color (no alternative material bands)
# room_wall_alternative_fns = {
#     t.Semantics.LivingRoom: (
#         "weighted_choice",
#         (2, "none"),
#         (1, "half"),
#         *([(v, k) for k, v in material_assignments.wall_plaster]),
#         # *([(v, k) for k, v in material_assignments.abstract_art]),
#     ),
#     t.Semantics.Bedroom: (
#         "weighted_choice",
#         (2, "none"),
#         (1, "half"),
#         *([(v, k) for k, v in material_assignments.wall_plaster]),
#         # *([(v, k) for k, v in material_assignments.abstract_art]),
#     ),
#     t.Semantics.Office: (
#         "weighted_choice",
#         (2, "none"),
#         (1, "half"),
#         *([(v, k) for k, v in material_assignments.wall_plaster]),
#         # *([(v, k) for k, v in material_assignments.abstract_art]),
#     ),
#     t.Semantics.OpenOffice: (
#         "weighted_choice",
#         (2, "none"),
#         (1, "half"),
#         *([(v, k) for k, v in material_assignments.wall_plaster]),
#         # *([(v, k) for k, v in material_assignments.abstract_art]),
#     ),
#     t.Semantics.FactoryOffice: (
#         "weighted_choice",
#         (2, "none"),
#         (1, "half"),
#         *([(v, k) for k, v in material_assignments.wall_plaster]),
#         # *([(v, k) for k, v in material_assignments.abstract_art]),
#     ),
#     t.Semantics.BreakRoom: (
#         "weighted_choice",
#         (2, "none"),
#         (1, "half"),
#         *([(v, k) for k, v in material_assignments.wall_plaster]),
#         # *([(v, k) for k, v in material_assignments.abstract_art]),
#     ),
# }
# room_wall_alternative_fns = defaultdict(
#     lambda: ("weighted_choice", (2, "none"), (0.5, "half")), room_wall_alternative_fns
# )
room_wall_alternative_fns = defaultdict(lambda: ("weighted_choice", (1, "none")))

room_no_curtain = {t.Semantics.Garage, t.Semantics.Warehouse}

pillar_rooms = {
    t.Semantics.LivingRoom,
    t.Semantics.Balcony,
    t.Semantics.DiningRoom,
    t.Semantics.Bedroom,
}


# TODO: add wall art and mirror to the walls
def room_walls(
    walls: list[bpy.types.Object], constants: RoomConstants, n_walls=3, material_seed=1
):
    walls = sorted(walls, key=lambda w: w.name)
    with FixedSeed(material_seed):
        wall_fns = list(
            weighted_sample(room_wall_fns[room_type(r.name)])() for r in walls
        )

        logger.debug(
            f"{room_walls.__name__} adding materials to {len(walls)=}, using {len(wall_fns)=}"
        )

        unique_wall_fns = list(dict.fromkeys(wall_fns))
        kwargs = dict(vertical=True, is_ceramic=True, alternating=False, shape="square")
        for wall_fn in sorted(unique_wall_fns, key=lambda f: f.__class__.__name__):
            shape = np.random.choice(["square", "rectangle", "hexagon"])
            kwargs = dict(vertical=True, alternating=False, shape=shape)
            rooms_ = [o for o, w in zip(walls, wall_fns) if w == wall_fn]
            indices = np.random.randint(0, n_walls, len(rooms_))
            for i in range(n_walls):
                rooms__ = [r for r, j in zip(rooms_, indices) if j == i]
                if wall_fn.__class__.__name__ == "Plaster":
                    for r in rooms__:
                        unwrap_normal(r, selection=None)
                if wall_fn.__class__.__name__ in [
                    "Brick",
                    "Concrete",
                    "Plaster",
                    "Wood",
                    "Metal",
                ]:
                    kwargs = {}
                surface.assign_material(rooms__, wall_fn(**kwargs))

        for wall, wall_fn in zip(walls, wall_fns):
            wall["surface_finish"] = (
                "tile" if wall_fn.__class__.__name__ == "Tile" else "paint"
            )

        for w in sorted(walls, key=lambda w: w.name):
            logger.debug(
                f"{room_walls.__name__} adding materials to {len(walls)=}, using {len(wall_fns)=}"
            )
            fn = rg(room_wall_alternative_fns[room_type(w.name)])
            match fn:
                case "none":
                    continue
                case "half":
                    continue  # disabled: skip half wall
                case _:
                    co = read_co(w)
                    u, v = read_edges(w).T
                    i = np.argmax(
                        read_edge_length(w) - 100 * (np.abs(co[u, -1] - co[v, -1]) > 0.1)
                    )
                    u_ = co[u[i]]
                    v_ = co[v[i]]
                    non_vertical = np.linalg.norm((co[u] - co[v])[:, :2], axis=-1) > 1e-2
                    directional = (
                        np.abs(np.cross((co[u] - co[v])[:, :2], (u_ - v_)[np.newaxis, :2]))
                        < 1e-4
                    )
                    collinear = (
                        np.abs(np.cross((co[u] - v_)[:, :2], (u_ - v_)[np.newaxis, :2]))
                        < 1e-4
                    )
                    collinear_ = (
                        np.abs(np.cross((co[u] - u_)[:, :2], (u_ - v_)[np.newaxis, :2]))
                        < 1e-4
                    )
                    aligned = non_vertical & directional & collinear & collinear_
                    with butil.ViewportMode(w, "EDIT"):
                        bm = bmesh.from_edit_mesh(w.data)
                        bm.faces.ensure_lookup_table()
                        alternative = np.zeros(len(bm.faces), dtype=int)
                        for f in bm.faces:
                            for e in f.edges:
                                if aligned[e.index]:
                                    alternative[f.index] = 1
                    write_attr_data(
                        w, "alternative", alternative, type="INT", domain="FACE"
                    )
                    mat_gen = fn()

                    if mat_gen.__class__.__name__ == "Plaster":
                        unwrap_normal(w, selection="alternative")

                    surface.assign_material(
                        w,
                        mat_gen(scale=log_uniform(0.5, 2.0), **kwargs),
                        selection="alternative",
                    )


# TODO: add wall art and mirror to the walls
def room_ceilings(ceilings, material_seed=1):
    ceilings = sorted(ceilings, key=lambda c: c.name)
    logger.debug(f"{room_ceilings.__name__} adding materials to {len(ceilings)=}")

    with FixedSeed(material_seed):
        ceiling_fns = list(
            weighted_sample(room_ceiling_fns[room_type(r.name)])() for r in ceilings
        )

        unique_ceiling_fns = list(dict.fromkeys(ceiling_fns))
        for ceiling_fn in sorted(unique_ceiling_fns, key=lambda f: f.__class__.__name__):
            rooms_ = [o for o, f in zip(ceilings, ceiling_fns) if f == ceiling_fn]
            if ceiling_fn.__class__.__name__ == "Plaster":
                for r in rooms_:
                    unwrap_normal(r, selection=None)

            surface.assign_material(rooms_, ceiling_fn())
            for r in rooms_:
                r["surface_finish"] = (
                    "tile" if ceiling_fn.__class__.__name__ == "Tile" else "paint"
                )


@gin.configurable
def room_floors(floors, n_floors=3, material_seed=1):
    floors = sorted(floors, key=lambda f: f.name)
    with FixedSeed(material_seed):
        floor_material_gens = []
        for r in floors:
            gen_class = weighted_sample(room_floor_fns[room_type(r.name)])()
            floor_material_gens.append(gen_class)

        logger.debug(
            f"{room_floors.__name__} adding materials to {len(floors)=}, using {len(floor_material_gens)=}"
        )

        unique_floor_fns = list(dict.fromkeys(floor_material_gens))
        for floor_fn in sorted(unique_floor_fns, key=lambda f: f.__class__.__name__):
            rooms_ = [o for o, f in zip(floors, floor_material_gens) if f == floor_fn]
            indices = np.random.randint(0, n_floors, len(rooms_))
            for i in range(n_floors):
                rooms__ = [r for r, j in zip(rooms_, indices) if j == i]
                if floor_fn.__class__.__name__ == "Plaster":
                    for r in rooms__:
                        unwrap_normal(r, selection=None)
                surface.assign_material(rooms__, floor_fn())


@gin.configurable
def populate_doors(
    placeholders: list[bpy.types.Object],
    constants: RoomConstants,
    state=None,
    n_doors=1,
    door_chance=1,
    # Flats of this era trim every opening with a stained architrave, so the
    # casing is the norm rather than an occasional extra.
    casing_chance=1.0,
    all_open=False,
    all_closed=False,
):
    # One factory so every door in this scene shares style and materials.
    factories = [
        random_door_factory()(np.random.randint(1e7), constants=constants)
        for _ in range(n_doors)
    ]

    logger.debug(
        f"{populate_doors.__name__} populating {len(placeholders)=} with {n_doors=} and {len(factories)=}"
    )

    indices = np.random.randint(0, len(factories), len(placeholders))
    col = butil.get_collection("unique_assets:doors")
    casing_col = butil.get_collection("unique_assets:door_casings")

    for i in trange(n_doors, desc="Placing doors"):
        factory = factories[i]
        factory.width = constants.door_width
        factory.height = constants.door_size
        factory.door_frame_style = "single_column"
        factory.door_orientation = "right"
        factory.door_frame_width = 0.04
        factory.shrink_width = 0.008
        factory.depth = min(0.045, constants.wall_thickness * 0.32)
        casing_factory = factory.casing_factory
        doors, casings = [], []
        for j in np.nonzero(indices == i)[0]:
            if uniform() > door_chance:
                continue
            if all_closed:
                rot_z = 0.0
            elif all_open:
                rot_z = uniform(0.93, 1.93)
            else:
                rot_p = uniform()
                if rot_p < 0.5:
                    rot_z = uniform(0, 0.1)
                elif rot_p < 0.7:
                    rot_z = uniform(0.93, 1.03)
                else:
                    rot_z = uniform(0, 1)
            rot_z *= np.pi / 2

            door = factory(int(j))
            door.parent = placeholders[j]
            y_sign = _placeholder_interior_y_sign(placeholders[j], state)
            door.location = (
                _door_cutout_local_x(factory, constants.door_width),
                y_sign * constants.wall_thickness / 2,
                -constants.door_size / 2,
            )
            door.rotation_euler[-1] = -rot_z
            doors.append(door)

            if uniform() < casing_chance:
                casing = casing_factory(int(j))
                casing.parent = placeholders[j]
                casing.location = 0, 0, -constants.door_size / 2
                casings.append(casing)

        factory.finalize_assets(doors)
        butil.put_in_collection(doors, col)
        casing_factory.finalize_assets(casings)
        butil.put_in_collection(casings, casing_col)


_WINDOW_HARDWARE_MATERIAL = None


def _window_hardware_material():
    """One metal for every window handle in the scene, matching door furniture."""
    global _WINDOW_HARDWARE_MATERIAL
    if _WINDOW_HARDWARE_MATERIAL is None:
        _WINDOW_HARDWARE_MATERIAL = weighted_sample(
            material_assignments.metal_neutral
        )()()
    return _WINDOW_HARDWARE_MATERIAL


@gin.configurable
def populate_windows(
    placeholders: list[bpy.types.Object], constants, state: state_def.State, n_windows=1
):
    # curtain / shutter have to be set on the factory: create_asset reads
    # self.curtain and self.shutter and ignores per-call kwargs. A flat handed
    # over before occupancy has bare glazing - no curtains hung yet, and no
    # louvred shutters blocking the daylight that lights the room.
    factories = [
        WindowFactory(np.random.randint(1e5), curtain=False, shutter=False)
        for _ in range(n_windows)
    ]

    logger.debug(
        f"{populate_windows.__name__} populating {len(placeholders)=} with {n_windows=} and {len(factories)=}"
    )

    col = butil.get_collection("unique_assets:windows")
    windows = []
    for j, cutter in enumerate(placeholders):
        cutter_dims = cutter.dimensions
        parent = state.objs[cutter.name].relations[0].target_name
        factory = factories[int_hash(parent) % n_windows]
        frame_thick = cutter_dims[1] * uniform(0.1, 0.2)
        dims = cutter_dims[0], cutter_dims[2], frame_thick

        window = factory(int(j), dimensions=dims, open=False)

        butil.put_in_collection(list(butil.iter_object_tree(window)), col)

        window.parent = cutter
        window.location[1] = -constants.wall_thickness / 2
        window.rotation_euler[1] = np.pi
        windows.append(window)
        factory.finalize_assets(windows)

        # Casement handle, screwed to a vertical stile on the room side. The
        # window fills the cutter in X and Z and its face normal is Y, so the
        # handle is turned to protrude along -Y (into the room) and set just
        # inside the opening edge, where the stile is - placing it by width
        # fraction alone left it floating out on the glass.
        handle = make_window_handle(
            int_hash((parent, j)), _window_hardware_material()
        )
        handle.parent = cutter
        handle.rotation_euler[2] = -np.pi / 2
        stile = 1.0 if uniform() < 0.5 else -1.0
        handle.location = (
            stile * max(cutter_dims[0] / 2 - 0.024, 0.0),
            # Sit the backplate on the room-side face of the frame. Offsetting
            # from the frame's centre plane instead buried all but 2 mm of the
            # handle inside the section.
            -constants.wall_thickness / 2 - frame_thick / 2 - 0.002,
            -cutter_dims[2] * uniform(0.02, 0.10),
        )
        butil.put_in_collection([handle], col)


def room_stairs(constants, state, rooms_meshed):
    col = butil.get_collection("unique_assets:staircases")

    if constants.n_stories == 1:
        return

    contours, doors = [], []
    for k, s in state.objs.items():
        if k.startswith(t.Semantics.StaircaseRoom.value):
            doors_ = [
                bpy.data.objects[l]
                for l, o in state.objs.items()
                if any(
                    r.relation == cl.CutFrom() and r.target_name == k
                    for r in o.relations
                )
                and l.startswith("door")
            ]
            p = shapely.Polygon(s.polygon)
            contour = shapely.simplify(
                p.buffer(-constants.wall_thickness / 2, join_style="mitre"), 0.1
            )
            for door in doors_:
                dw = constants.door_width
                box = shapely.box(-dw / 2, -dw * 1.5, dw / 2, dw * 1.5)
                box = shapely.affinity.translate(
                    shapely.affinity.rotate(box, door.rotation_euler[-1]),
                    *door.location,
                )
                contour = contour.difference(box)
            doors.append(doors_)
            contours.append(contour)

    geoms = []
    for c, c_ in zip(contours[:-1], contours[1:]):
        geoms.append(c.intersection(c_).buffer(0))

    placeholders, offsets, fns = [], [], []
    for _ in trange(200, desc="Generating staircases: "):
        butil.delete(placeholders)
        fns = [
            random_staircase_factory()(np.random.randint(1e7), False, constants)
            for _ in geoms
        ]
        placeholders, mlss, lower, upper = [], [], [], []
        for j, fn in enumerate(fns):
            ph = fn.create_placeholder(i=np.random.randint(1e7))
            placeholders.append(ph)
            polygon = shapely.intersection_all(
                list(
                    shapely.affinity.translate(geoms[j], -x, -y)
                    for x in [ph.bound_box[0][0], ph.bound_box[-1][0]]
                    for y in [ph.bound_box[0][1], ph.bound_box[-1][1]]
                )
            )
            mlss.append(
                polygon.exterior
                if polygon.geom_type == "Polygon"
                else shapely.MultiLineString([p.exterior for p in polygon.geoms])
            )
            x, y, z = read_co(ph).T
            lower.append((x[z < constants.wall_height], y[z < constants.wall_height]))
            upper.append((x[z >= constants.wall_height], y[z >= constants.wall_height]))
        if any(p.is_empty for p in mlss):
            continue
        for _ in range(50):
            offsets = []
            for j, mls in enumerate(mlss):
                b = mls.bounds
                for _ in range(50):
                    x = uniform(b[0], b[2])
                    y = uniform(b[1], b[3])
                    p = Point(x, y)
                    projected = nearest_points(mls, p)[0]
                    if (
                        max(np.abs(p.x - projected.x), np.abs(p.y - projected.y))
                        < constants.staircase_snap
                    ):
                        p = projected
                        coords = (
                            np.concatenate([ls.coords for ls in mls.geoms])
                            if mls.geom_type == "MultiLineString"
                            else mls.coords
                        )
                        projected = nearest_points(
                            shapely.MultiPoint(coords), Point(x, y)
                        )[0]
                        if (
                            max(np.abs(p.x - projected.x), np.abs(p.y - projected.y))
                            <= constants.staircase_snap
                        ):
                            p = projected
                    x, y = p.x, p.y
                    placeholders[j].location = (
                        x,
                        y,
                        j * constants.wall_height + constants.wall_thickness / 2,
                    )
                    contains_lower = shapely.contains_xy(
                        contours[j], lower[j][0] + x, lower[j][1] + y
                    ).all()
                    contains_upper = shapely.contains_xy(
                        contours[j + 1], upper[j][0] + x, upper[j][1] + y
                    ).all()
                    lower_valid = fns[j].valid_contour((x, y), contours[j], doors[j])
                    upper_valid = fns[j].valid_contour(
                        (x, y), contours[j + 1], doors[j + 1], False
                    )
                    if (
                        contains_lower
                        and contains_upper
                        and lower_valid
                        and upper_valid
                    ):
                        offsets.append((x, y))
                        break
            if len(offsets) == len(geoms):
                ts = list(
                    trimesh.convex.convex_hull(
                        obj2trimesh(ph).apply_transform(
                            translation_matrix([*o, constants.wall_height * j])
                        )
                    )
                    for j, (ph, o) in enumerate(zip(placeholders, offsets))
                )
                if all(t.intersection(t_).is_empty for t, t_ in zip(ts[:-1], ts[1:])):
                    break
        if len(offsets) == len(geoms):
            break
    butil.delete(placeholders)
    if len(offsets) != len(geoms):
        return
    for j, fn in enumerate(tqdm(fns)):
        s = fn(i=np.random.randint(1e7))
        fn.finalize_assets(s)
        butil.apply_transform(s, True)
        s.location = (
            *offsets[j],
            j * constants.wall_height + constants.wall_thickness / 2,
        )

        mesh, mesh_ = None, None
        for m in rooms_meshed:
            if room_type(m.name) == t.Semantics.StaircaseRoom:
                level = room_level(m.name)
                if level == j + 1:
                    mesh = m
                elif level == j:
                    mesh_ = m
        if mesh is None or mesh_ is None:
            butil.put_in_collection(s, col)
            continue
        cutter = fn.create_cutter(i=np.random.randint(1e7))
        cutter.location = (
            *offsets[j],
            j * constants.wall_height + constants.wall_thickness / 2,
        )
        butil.modify_mesh(
            mesh,
            "BOOLEAN",
            object=cutter,
            operation="DIFFERENCE",
            use_self=True,
            use_hole_tolerant=True,
        )
        butil.modify_mesh(
            mesh_,
            "BOOLEAN",
            object=cutter,
            operation="DIFFERENCE",
            use_self=True,
            use_hole_tolerant=True,
        )
        butil.delete(cutter)

        m = deep_clone_obj(mesh)
        m.location = -offsets[j][0], -offsets[j][1], 0
        butil.apply_transform(m, True)
        g = fns[j].make_guardrail(m)
        g.location = s.location
        g.location[-1] += constants.wall_height
        butil.put_in_collection([s, g], col)


def room_pillars(walls: list[bpy.types.Object], constants: RoomConstants):
    col = butil.get_collection("unique_assets:pillars")
    for wall in tqdm(walls):
        if room_type(wall.name) not in pillar_rooms:
            continue

        factory = PillarFactory(np.random.randint(1e7), False, constants)
        interior = tagging.extract_tagged_faces(wall, {t.Subpart.Interior})
        dissolve_limited(interior)
        cos = []
        with butil.ViewportMode(interior, "EDIT"):
            bm = bmesh.from_edit_mesh(interior.data)
            for e in bm.edges:
                u, v = e.verts
                is_angled = np.pi * 0.1 < e.calc_face_angle(0) % np.pi < np.pi * 0.9
                is_long = e.calc_length() > constants.wall_height * 0.8
                is_vertical = (
                    np.abs(u.co[-1] - v.co[-1]) / (e.calc_length() + 1e-6) > 0.9
                )
                if is_long and is_vertical and is_angled:
                    cos.append(u.co)
        if len(cos) == 0:
            butil.delete(interior)
            continue
        cos = np.array(cos)
        cos += np.array(interior.location)[np.newaxis, :]

        joins = [
            read_co(o) + np.array([o.location])
            for o in butil.get_collection("staircases").objects
        ] + [
            read_co(o) + np.array([o.location])
            for o in butil.get_collection("doors").objects
        ]
        if len(joins) == 0:
            joins = np.zeros((1, 3))
        placeholders = np.concatenate(joins)
        cos[:, -1] = wall.location[-1] + constants.wall_thickness / 2
        cos = cos[
            np.min(
                np.linalg.norm(cos[:, np.newaxis] - placeholders[np.newaxis], axis=-1),
                -1,
            )
            > constants.door_width / 2 + constants.wall_thickness
        ]
        interior_xy = None
        if len(interior.data.vertices):
            ico = read_co(interior) + np.array(interior.location)
            interior_xy = ico[:, :2].mean(axis=0)
        for co in cos:
            obj = factory(int(np.random.randint(1e7)))
            factory.finalize_assets([obj])
            loc = np.array(co, dtype=float)
            if interior_xy is not None:
                delta = interior_xy - loc[:2]
                nrm = np.linalg.norm(delta)
                if nrm > 1e-4:
                    loc[:2] = loc[:2] + delta / nrm * (factory.width * 0.55)
            obj.location = loc
            obj.location[-1] = (
                room_level(wall.name) * constants.wall_height
                + constants.wall_thickness / 2
            )
            butil.put_in_collection(obj, col)
        butil.delete(interior)


@gin.configurable
def room_ceiling_beams(
    ceilings,
    constants=None,
    enabled=True,
    depth=("uniform", 0.22, 0.40),
    width=("uniform", 0.16, 0.26),
    perimeter_chance=0.85,
    span_beam_chance=0.55,
    min_room_span=3.2,
    material_seed=1,
):
    """Downstand beams under the ceiling slab.

    Flats of this construction have the beams showing: a band around the
    perimeter where the slab meets the loadbearing walls, and a beam across the
    room when the span is long enough to need one. Without them the ceiling
    reads as a featureless plane, which is one of the strongest synthetic tells.
    """
    if not enabled:
        logger.info("Skipping ceiling beams")
        return
    if constants is None:
        constants = RoomConstants()

    col = butil.get_collection("unique_assets:ceiling_beams")
    with FixedSeed(material_seed):
        for ceiling in sorted(ceilings, key=lambda c: c.name):
            try:
                poly = obj2polygon(ceiling)
            except Exception as e:  # degenerate ceilings are not worth beaming
                logger.warning("ceiling beam skipped for %s: %s", ceiling.name, e)
                continue
            if poly.is_empty or poly.area < 1.0:
                continue
            # obj2polygon reads local coordinates and split_rooms re-centres each
            # ceiling's origin, so the footprint has to be carried back into world
            # space or every room's beams are built on top of each other at (0,0).
            loc = ceiling.matrix_world.translation
            poly = shapely.affinity.translate(poly, loc.x, loc.y)
            polys = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]

            d = rg(depth)
            w = rg(width)
            z_top = loc.z
            parts = []
            for sub in polys:
                if sub.area < 1.0:
                    continue
                parts += _beam_solids(sub, w, d, z_top, perimeter_chance,
                                      span_beam_chance, min_room_span)
            if not parts:
                continue
            obj = join_objects(parts) if len(parts) > 1 else parts[0]
            obj.name = f"{ceiling.name.split('.')[0]}.beams"
            mats = [m for m in ceiling.data.materials if m is not None]
            if mats:
                surface.assign_material(obj, mats[0])
            butil.put_in_collection(obj, col)


def _beam_solids(poly, w, d, z_top, perimeter_chance, span_beam_chance, min_span):
    """Ring under the perimeter plus, on a long room, one beam across it."""
    from infinigen.assets.utils.shapes import buffer, polygon2obj

    solids = []

    def _extrude(flat_poly):
        if flat_poly.is_empty or flat_poly.area < 1e-4:
            return
        for sub in (
            flat_poly.geoms if flat_poly.geom_type == "MultiPolygon" else [flat_poly]
        ):
            if sub.area < 1e-4:
                continue
            o = polygon2obj(sub, z=z_top - d)
            butil.modify_mesh(o, "SOLIDIFY", thickness=d, offset=1)
            solids.append(o)

    if uniform() < perimeter_chance:
        _extrude(poly.difference(buffer(poly, -w)))

    x0, y0, x1, y1 = poly.bounds
    dx, dy = x1 - x0, y1 - y0
    if max(dx, dy) > min_span and uniform() < span_beam_chance:
        # Beam crosses the short way, as it spans between the long walls.
        if dx >= dy:
            c = uniform(x0 + dx * 0.35, x0 + dx * 0.65)
            band = shapely.box(c - w / 2, y0 - 1, c + w / 2, y1 + 1)
        else:
            c = uniform(y0 + dy * 0.35, y0 + dy * 0.65)
            band = shapely.box(x0 - 1, c - w / 2, x1 + 1, c + w / 2)
        _extrude(poly.intersection(band))

    return solids


@gin.configurable
def room_cable_trunks(walls, constants=None, ceilings=None, enabled=False):
    """Full-length PVC trunking along wall–ceiling lines, branched to lights."""
    if not enabled:
        logger.info("Skipping cable trunks")
        return
    from infinigen.assets.objects.wall_decorations.cable_trunk import (
        install_room_cable_trunks,
    )

    install_room_cable_trunks(list(walls), ceilings=list(ceilings or []))


WALL_DEFECT_FACTORIES = {
    "CrackPlaneFactory",
    "PaintPeelPlaneFactory",
    "WallBubblePlaneFactory",
    "PaintRunPlaneFactory",
    "PaintPatchPlaneFactory",
    "SpallingPlaneFactory",
    "SpallingPlugPlaneFactory",
    "OpenWiringPlaneFactory",
    "WeakLeakStainPlaneFactory",
}
CEILING_DEFECT_FACTORIES = {
    "CeilingCrackPlaneFactory",
    "CeilingPeelFactory",
}


def _room_stem(name):
    if name.endswith((".wall", ".ceiling", ".floor")):
        return name.rsplit(".", 1)[0]
    return name.split(".")[0]


def _is_tiled_surface(obj):
    if obj is None:
        return False
    if obj.get("surface_finish") == "tile":
        return True
    mats = []
    if getattr(obj, "active_material", None) is not None:
        mats.append(obj.active_material)
    data = getattr(obj, "data", None)
    if data is not None:
        mats.extend(m for m in getattr(data, "materials", []) if m is not None)
    for mat in mats:
        n = (mat.name or "").lower()
        if "wood" in n:
            continue
        if n.endswith("_tile") or "_tile_" in n or n.startswith("tile"):
            return True
    return False


def strip_defects_on_tiled_surfaces(walls, ceilings, state):
    """Drop paint/crack defects sitting on tiled bathroom/kitchen finishes."""
    tiled_walls = {_room_stem(w.name) for w in (walls or []) if _is_tiled_surface(w)}
    tiled_ceils = {
        _room_stem(c.name) for c in (ceilings or []) if _is_tiled_surface(c)
    }
    if not tiled_walls and not tiled_ceils:
        return 0

    to_delete = []
    for os in state.objs.values():
        gen = getattr(os, "generator", None)
        if gen is None:
            continue
        gname = gen.__class__.__name__
        hosts = [_room_stem(r.target_name) for r in os.relations]
        drop = False
        if gname in WALL_DEFECT_FACTORIES:
            drop = any(h in tiled_walls for h in hosts)
        elif gname in CEILING_DEFECT_FACTORIES:
            drop = any(h in tiled_ceils for h in hosts)
        if drop and os.obj is not None:
            to_delete.append(os.obj)
            os.obj = None

    if not to_delete:
        return 0

    logger.info(
        "Removing %s defects from tiled surfaces (rooms walls=%s ceilings=%s)",
        len(to_delete),
        sorted(tiled_walls),
        sorted(tiled_ceils),
    )
    deleted = set(to_delete)
    try:
        from infinigen.core.constraints.example_solver import populate as pop

        for attr in (
            "deferred_wall_bubble_finalize",
            "deferred_paint_run_finalize",
            "deferred_paint_patch_finalize",
        ):
            seq = getattr(pop, attr, None)
            if seq:
                seq[:] = [(g, o) for g, o in seq if o not in deleted]
    except Exception as exc:
        logger.warning("Could not prune deferred defect finalize lists: %s", exc)

    butil.delete(to_delete)
    return len(to_delete)
