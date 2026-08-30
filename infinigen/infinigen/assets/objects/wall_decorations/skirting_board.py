# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo, Lingjie Mei, Alexander Raistrick

import logging

import bpy
from mathutils import Vector
import numpy as np
import shapely
from numpy.random import randint, uniform
from shapely.geometry import Polygon
from shapely.ops import unary_union

import infinigen.core.util.blender as butil
from infinigen.assets import colors
from infinigen.assets.materials.plastic import plastic_rough
from infinigen.assets.materials.wood.wood import (
    sample_interior_wood_color,
    shader_wood,
)
from infinigen.assets.utils.decorate import (
    read_co,
)
from infinigen.assets.utils.draw import bezier_curve
from infinigen.assets.utils.object import new_plane
from infinigen.assets.utils.shapes import obj2polygon
from infinigen.core import surface, tagging
from infinigen.core import tags as t
from infinigen.core.constraints.example_solver.room.base import room_level
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed

logger = logging.getLogger(__name__)


@node_utils.to_nodegroup(
    "nodegroup_make_skirting_board_001", singleton=False, type="GeometryNodeTree"
)
def nodegroup_make_skirting_board(nw: NodeWrangler, control_points):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketCollection", "Parent", None),
            ("NodeSocketFloat", "Thickness", 0.0300),
            ("NodeSocketFloat", "Height", 0.1500),
            ("NodeSocketFloat", "Resolution", 0.0050),
            ("NodeSocketBool", "Is Ceiling", False),
        ],
    )

    collection_info = nw.new_node(
        Nodes.CollectionInfo, input_kwargs={"Collection": group_input.outputs["Parent"]}
    )

    mesh = nw.new_node(
        Nodes.RealizeInstances, input_kwargs={"Geometry": collection_info}
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["Thickness"],
            "Height": group_input.outputs["Height"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply, "Y": multiply_1}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": quadrilateral, "Translation": combine_xyz},
    )

    resample_curve_1 = nw.new_node(
        Nodes.ResampleCurve,
        input_kwargs={
            "Curve": transform_geometry,
            "Length": group_input.outputs["Resolution"],
        },
        attrs={"mode": "LENGTH"},
    )

    position = nw.new_node(Nodes.InputPosition)

    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={"Vector": position})

    greater_than = nw.new_node(
        Nodes.Compare, input_kwargs={0: separate_xyz.outputs["X"]}
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={"Value": separate_xyz.outputs["Y"], 1: multiply_2, 2: 0.0000},
    )

    float_curve = nw.new_node(
        Nodes.FloatCurve, input_kwargs={"Value": map_range.outputs["Result"]}
    )
    node_utils.assign_curve(float_curve.mapping.curves[0], control_points)

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: float_curve, 1: group_input.outputs["Thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_3, "Y": separate_xyz.outputs["Y"]}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": resample_curve_1,
            "Selection": greater_than,
            "Position": combine_xyz_1,
        },
    )

    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            0: group_input.outputs["Is Ceiling"],
            1: (-1.0000, 1.0000, 1.0000),
            2: (-1.0000, -1.0000, -1.0000),
        },
        attrs={"input_type": "VECTOR"},
    )

    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_position, "Scale": switch},
    )

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": mesh,
            "Profile Curve": transform_geometry_1,
            "Fill Caps": True,
        },
    )

    set_shade_smooth = nw.new_node(
        Nodes.SetShadeSmooth,
        input_kwargs={"Geometry": curve_to_mesh_1, "Shade Smooth": False},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": set_shade_smooth},
        attrs={"is_active_output": True},
    )


def apply_skirtingboard(
    nw: NodeWrangler, contour, is_ceiling=False, seed=None, thickness=0.02
):
    # Code generated using version 2.6.5 of the node_transpiler

    # TODO: randomize style / size / materials
    if seed is None:
        seed = randint(0, 10000)
    with FixedSeed(seed):
        thickness = uniform(0.02, 0.05)
        height = uniform(0.08, 0.15)
        # Cornices are painted with the ceiling; floor skirting in these flats is
        # usually stained to match the door architraves, and only sometimes white.
        stained = (not is_ceiling) and uniform() < 0.75
        color = (
            sample_interior_wood_color() if stained else hsv2rgba(colors.white_hsv())
        )
        roughness = uniform(0.35, 0.6) if stained else uniform(0.5, 1.0)
        n_peaks = randint(1, 4)
        start_y = uniform(0.0, 0.5)
        mid_x = uniform(0.2, 0.8)
        peak_xs = np.sort(uniform(0.0, mid_x, size=n_peaks))
        peak_ys = np.sort(uniform(start_y, 1.0, size=n_peaks))
        control_points = [(0.0000, start_y)]
        control_points += [(x, y) for x, y in zip(peak_xs, peak_ys)]
        control_points += [(mid_x, 1.0000), (1.0000, 1.0000)]

    makeskirtingboard = nw.new_node(
        nodegroup_make_skirting_board(control_points=control_points).name,
        input_kwargs={
            "Parent": contour,
            "Resolution": 0.0010,
            "Thickness": thickness,
            "Height": height,
            "Is Ceiling": is_ceiling,
        },
    )

    makeskirtingboard = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": makeskirtingboard,
            "Material": (
                surface.shaderfunc_to_material(shader_wood, color=color)
                if stained
                else surface.shaderfunc_to_material(
                    plastic_rough.shader_rough_plastic,
                    base_color=color,
                    roughness=roughness,
                    displacement_scale=uniform(2.5, 4.0),
                )
            ),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": makeskirtingboard},
        attrs={"is_active_output": True},
    )


def make_skirtingboard_contour(objs: list[bpy.types.Object], tag: t.Subpart, constants):
    # make the outline curve

    assert len(objs) > 0

    objs = [
        tagging.extract_tagged_faces(o, {tag, t.Subpart.Visible}, nonempty=True)
        for o in list(objs)
    ]

    all_polys = []
    all_zs = []
    for floor_pieces in objs:
        all_polys.append(obj2polygon(floor_pieces))
        all_zs.append(read_co(floor_pieces)[:, -1] + floor_pieces.location[-1])

    floor_z = np.mean(np.concatenate(all_zs))

    boundary = (
        unary_union(all_polys)
        .buffer(0.05, join_style="mitre")
        .buffer(-0.05, join_style="mitre")
    )

    if isinstance(boundary, Polygon):
        boundaries = [boundary]
    else:
        boundaries = boundary.geoms

    contours = []

    # Openings are taken out of the contour here, in plan, rather than by
    # differencing the finished board in 3D. The swept board is a thin closed
    # shell and Blender's exact solver returns an empty mesh for it often
    # enough to matter - against a storey-wide board that only lost a stretch,
    # but against a single room's board it took the whole thing (bedroom 3042
    # verts -> 0, kitchen 2248 -> 0), which is why per-room boards vanished.
    # Subtracting rectangles from a ring is exact, and much faster.
    band = 0.15 if tag == t.Subpart.Ceiling else 0.0
    openings = _opening_footprints(constants, floor_z - band, floor_z + band)
    cut = unary_union(openings) if openings else None

    for b in boundaries:
        for lr, rev in [(b.exterior, False)] + [(i, True) for i in b.interiors]:
            for o in linear_ring2curve(lr, constants, rev, cut):
                contours.append(o)
                o.location[-1] += floor_z
                butil.apply_transform(o, True)
    butil.delete(objs)
    return contours


def _world_z_range(obj):
    zs = [(obj.matrix_world @ Vector(c)).z for c in obj.bound_box]
    return min(zs), max(zs)


def _opening_footprints(constants, z_lo, z_hi, reveal=0.02):
    """Plan-view rectangles of the openings that cross a given height band.

    Built from the portal cutters directly, in world XY. The cutter box is
    about a metre deep so it can punch through the wall; used at that depth it
    would also take the board off the walls returning either side of the
    opening, so it is clamped to just over the wall thickness. The rectangle is
    exact rather than an AABB, which matters on a wall that is not axis
    aligned.
    """
    boxes = []
    for p in butil.get_collection("placeholders:portal_cutters").objects:
        if not p.name.startswith(("door", "entrance")):
            continue
        z0, z1 = _world_z_range(p)
        # Unplaced cutters still sit at the origin; a real opening spans the
        # band the board occupies.
        if z0 > z_lo + 0.15 or z1 < z_hi - 0.15:
            continue

        scale = p.matrix_world.to_scale()
        half_x = max((abs(v[0]) for v in p.bound_box), default=0.0) * abs(scale.x)
        half_y = max((abs(v[1]) for v in p.bound_box), default=0.0) * abs(scale.y)
        if half_x < 0.05 or half_y < 1e-6:
            continue

        sx = half_x + reveal
        sy = min(half_y, constants.wall_thickness * 0.5 + 0.06)
        rot = p.rotation_euler[2]
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        loc = p.matrix_world.translation
        corners = [
            (
                loc.x + a * sx * cos_r - b * sy * sin_r,
                loc.y + a * sx * sin_r + b * sy * cos_r,
            )
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
        boxes.append(shapely.Polygon(corners))
    return boxes


def make_skirting_board(constants, objs, tag, joined=False, keep_rooms=None):
    """One board per room (not a storey-wide union).

    A single joined contour follows the apartment outline, so internal walls
    get no skirting and the board cannot be hidden per room. Per-room boards
    sit on each room's own walls and can be culled with hide_other_rooms.
    """
    if keep_rooms is not None:
        objs = [
            o
            for o in objs
            if any(k.split(".")[0] in o.name for k in keep_rooms)
        ]
    if not objs:
        return

    if joined:
        seqs = list(
            [o for o in objs if room_level(o.name.split(".")[0]) == i]
            for i in range(constants.n_stories)
        )
    else:
        seqs = [[o] for o in objs]

    for s in seqs:
        if not s:
            continue
        logger.debug(f"make_skirting_board for {len(s)=} {tag=}")

        try:
            contours = make_skirtingboard_contour(s, tag, constants)
        except shapely.errors.GEOSException as e:
            logger.warning(
                f"make_skirting_board({s=}, {tag=}) failed with {e}, skipping"
            )
            continue

        obj = new_plane()
        stem = s[0].name.split(".")[0]
        obj.name = f"{stem}.skirtingboard_{tag.value}"

        col = butil.put_in_collection(contours, "contour")
        kwargs = {
            "contour": col,
            "seed": np.random.randint(1e7),
            "is_ceiling": tag == t.Subpart.Ceiling,
        }
        surface.add_geomod(obj, apply_skirtingboard, apply=True, input_kwargs=kwargs)

        butil.delete_collection(col)
        col = butil.get_collection("skirting")
        butil.put_in_collection(obj, col)


def linear_ring2curve(ring, constants, reversed=False, cut=None):
    """The ring, minus the doorways it runs through, as one curve per run.

    This used to drop any segment whose length happened to fall within 2 cm of
    wall_thickness or door_width, as a way of guessing where the doorways were.
    It guesses wrong: an ordinary ~0.9 m wall run is indistinguishable from a
    door opening by length alone, so stretches of board went missing for no
    visible reason. `cut` is the real openings, taken from the portal cutters.
    """
    coords = ring.coords
    if shapely.is_ccw(ring) == reversed:
        coords = coords[::-1]

    line = shapely.LineString(coords)
    if cut is not None and not cut.is_empty:
        line = line.difference(cut)
    parts = (
        list(line.geoms) if line.geom_type == "MultiLineString" else [line]
    )

    curves = []
    for part in parts:
        if part.is_empty or part.geom_type != "LineString":
            continue
        pts = np.array(part.coords)
        if len(pts) < 2:
            continue
        x, y = pts.T
        curves.append(bezier_curve((x, y, 0), list(np.arange(len(x))), 1, False))
    return curves
