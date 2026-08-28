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
from infinigen.assets.utils.object import join_objects, new_plane
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

    for b in boundaries:
        lr = b.exterior
        o = linear_ring2curve(lr, constants)
        contours.append(o)
        o.location[-1] += floor_z
        butil.apply_transform(o, True)
        for lr in b.interiors:
            o = linear_ring2curve(lr, constants, True)
            contours.append(o)
            o.location[-1] += floor_z
            butil.apply_transform(o, True)
    butil.delete(objs)
    return contours


def _world_z_range(obj):
    zs = [(obj.matrix_world @ Vector(c)).z for c in obj.bound_box]
    return min(zs), max(zs)


def _cut_floor_openings(skirting, floor_tol=0.15):
    """Stop the board at every doorway.

    Only entrances used to be cut, so skirting ran straight across interior
    doorways. The opening is widened to the architrave before differencing so
    the board butts into the casing rather than disappearing behind it - and the
    operand is a widened copy of the cutter box, since the casing mesh is itself
    a boolean result and makes an unreliable operand.
    """
    cutters = butil.get_collection("placeholders:portal_cutters").objects
    casings = butil.get_collection("unique_assets:door_casings").objects
    skirt_z0, _ = _world_z_range(skirting)

    for p in list(cutters):
        if not p.name.startswith(("door", "entrance")):
            continue
        # Cutters left sitting at the origin are unplaced; a placed opening
        # starts level with the floor the skirting sits on.
        if abs(_world_z_range(p)[0] - skirt_z0) > floor_tol:
            continue

        # How far past the opening the architrave reaches, so the board butts
        # into the casing instead of vanishing behind it. bound_box is local, so
        # fold in each object's world scale, and hard-clamp the widening: a
        # near-zero half-width once produced a runaway factor here and the
        # resulting boolean operand exhausted memory mid-solve.
        sx = max(abs(p.matrix_world.to_scale().x), 1e-6)
        half = max((abs(v[0]) for v in p.bound_box), default=0.0) * sx
        if half < 0.05:
            continue
        trim = half
        for c in casings:
            if c.parent is not p or not len(c.data.vertices):
                continue
            csx = max(abs(c.matrix_world.to_scale().x), 1e-6)
            trim = max(trim, max(abs(v[0]) for v in c.bound_box) * csx)

        ratio = trim / half
        if ratio > 1.6:
            logger.warning(
                "%s: architrave/opening ratio %.1f is implausible (half=%.3f "
                "trim=%.3f); clamping so the boolean operand stays sane",
                p.name,
                ratio,
                half,
                trim,
            )
        operand = butil.copy(p)
        operand.scale[0] *= min(max(ratio, 1.0), 1.6)
        # A convex box operand needs neither self-intersection handling nor hole
        # tolerance, and both are costly: this runs once per opening over a
        # whole storey's skirting, where the old entrance-only cut ran once.
        butil.modify_mesh(
            skirting, "BOOLEAN", object=operand, operation="DIFFERENCE"
        )
        butil.delete(operand)


def _cut_ceiling_entrances(skirting, constants):
    """Cornice keeps the old entrance-only behaviour: interior doors stop below it."""
    for p in butil.get_collection("placeholders:portal_cutters").objects:
        if (
            p.name.startswith("entrance")
            and int(p.location[-1] / constants.wall_height - 1 / 2) == 0
        ):
            p.location[-1] -= constants.wall_height / 2
            butil.modify_mesh(
                skirting,
                "BOOLEAN",
                object=p,
                operation="DIFFERENCE",
                use_self=True,
                use_hole_tolerant=True,
            )
            p.location[-1] += constants.wall_height / 2


def make_skirting_board(constants, objs, tag, joined=True):
    if joined:
        seqs = list(
            [o for o in objs if room_level(o.name.split(".")[0]) == i]
            for i in range(constants.n_stories)
        )
    else:
        seqs = [[o] for o in objs]

    for s in seqs:
        logger.debug(f"make_skirting_board for {len(objs)=} {tag=}")

        try:
            contours = make_skirtingboard_contour(s, tag, constants)
        except shapely.errors.GEOSException as e:
            logger.warning(
                f"make_skirting_board({objs=}, {tag=}) failed with {e}, skipping"
            )
            return

        obj = new_plane()
        obj.name = "skirtingboard_" + tag.value

        col = butil.put_in_collection(contours, "contour")
        kwargs = {
            "contour": col,
            "seed": np.random.randint(1e7),
            "is_ceiling": tag == t.Subpart.Ceiling,
        }
        surface.add_geomod(obj, apply_skirtingboard, apply=True, input_kwargs=kwargs)

        if tag == t.Subpart.Ceiling:
            _cut_ceiling_entrances(obj, constants)
        else:
            _cut_floor_openings(obj)
        butil.delete_collection(col)
        col = butil.get_collection("skirting")
        butil.put_in_collection(obj, col)


def linear_ring2curve(ring, constants, reversed=False):
    coords = ring.coords
    if shapely.is_ccw(ring) == reversed:
        coords = coords[::-1]
    coords = np.array(coords)
    lengths = np.linalg.norm(coords[:-1] - coords[1:], axis=-1)
    invalid = np.sort(
        np.nonzero(
            (np.abs(lengths - constants.wall_thickness) < 0.02)
            | (np.abs(lengths - constants.door_width) < 0.02)
        )[0]
    )
    ranges = -1, *invalid, len(coords)
    curves = []
    for l, r in zip(ranges[:-1], ranges[1:]):
        x, y = np.array(coords[l + 1 : r + 1]).T
        if len(x) > 1:
            curves.append(bezier_curve((x, y, 0), list(np.arange(len(x))), 1, False))
    return join_objects(curves)
