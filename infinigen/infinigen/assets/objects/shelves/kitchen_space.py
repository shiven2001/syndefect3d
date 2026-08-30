# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo, Stamatis Alexandropoulos

import bpy
from mathutils import Vector
from numpy.random import uniform

from infinigen.assets.materials.ceramic.marble import shader_marble
from infinigen.assets.objects.appliances.cooktop import CooktopFactory
from infinigen.assets.objects.shelves.kitchen_cabinet import KitchenCabinetFactory
from infinigen.assets.objects.tables.table_top import nodegroup_generate_table_top
from infinigen.assets.objects.wall_decorations.range_hood import RangeHoodFactory
from infinigen.assets.utils.object import new_bbox
from infinigen.core import surface, tagging
from infinigen.core import tags as t
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


def nodegroup_tag_cube(nw: NodeWrangler):
    # Code generated using version 2.6.4 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    index = nw.new_node(Nodes.Index)

    equal = nw.new_node(
        Nodes.Compare,
        input_kwargs={2: index, 3: 5},
        attrs={"data_type": "INT", "operation": "EQUAL"},
    )

    cube = tagging.tag_nodegroup(
        nw, group_input.outputs["Geometry"], t.Subpart.SupportSurface, selection=equal
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": cube},
        attrs={"is_active_output": True},
    )


def geometry_nodes_add_cabinet_top(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 0.0500

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Min"]}
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz.outputs["X"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract, 1: 1.4140},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Y"], 1: separate_xyz.outputs["Y"]},
        attrs={"operation": "SUBTRACT"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_1, 1: subtract},
        attrs={"operation": "DIVIDE"},
    )

    generatetabletop = nw.new_node(
        nodegroup_generate_table_top().name,
        input_kwargs={
            "Thickness": value,
            "N-gon": 4,
            "Profile Width": multiply,
            "Aspect Ratio": divide,
            "Fillet Ratio": 0.0100,
            "Fillet Radius Vertical": 0.0100,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": generatetabletop,
            "Material": surface.shaderfunc_to_material(shader_marble),
        },
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    divide_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: add, 1: 2.0000}, attrs={"operation": "DIVIDE"}
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": bounding_box.outputs["Max"]}
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": divide_1, "Z": separate_xyz_2.outputs["Z"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": set_material, "Translation": combine_xyz},
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [group_input.outputs["Geometry"], transform_geometry]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


def geometry_node_to_tagged_bbox(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler
    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": bounding_box, "Scale": (0.9700, 0.9700, 1.000)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


def geometry_node_to_bbox(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler
    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    transform_geometry = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": bounding_box, "Scale": (0.9700, 0.9700, 1.000)},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_geometry},
        attrs={"is_active_output": True},
    )


class KitchenSpaceFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None, island=False):
        super(KitchenSpaceFactory, self).__init__(factory_seed, coarse=coarse)

        with FixedSeed(factory_seed):
            if dimensions is None:
                dimensions = Vector(
                    (
                        uniform(0.7, 1),
                        # A single 5 m run is a mansion's kitchen, and in a
                        # flat it monopolises the only wall long enough to take
                        # it - which left no room for the oven, so the solver
                        # placed the oven and gave up on the counter entirely.
                        uniform(1.6, 3.0),
                        # Wall units stop under the downstand beam. The beam
                        # soffit sits at ceiling - 0.24..0.32, so from a
                        # finished floor that is 2.39 m of headroom at worst;
                        # 2.3-2.5 drove the cabinet tops and the hood straight
                        # into the beam. _clamp_kitchen_runs_under_beams takes
                        # up whatever a low ceiling still leaves over.
                        uniform(2.15, 2.35),
                    )
                )

            self.island = island
            if self.island:
                dimensions.x *= uniform(1.5, 2)

            self.dimensions = dimensions

            self.params = self.sample_parameters(dimensions)

    def sample_parameters(self, dimensions):
        self.cabinet_bottom_height = uniform(0.8, 1.0)
        self.cabinet_top_height = uniform(0.8, 1.0)

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        x, y, z = self.dimensions
        # Exactly the asset's own depth. The 1.08 made the placeholder 8%
        # deeper than the run it stands for, and since the solver seats the
        # placeholder against the wall the cabinets ended up floating 4% of
        # their depth - 30 to 40 mm - off the plaster.
        box = new_bbox(-x / 2, x / 2, 0, y, 0, self.cabinet_bottom_height + 0.13)
        surface.add_geomod(box, nodegroup_tag_cube, apply=True)

        if not self.island:
            # Keep the wall-unit box clear of the base box. With a lower run
            # (z 2.15) and a tall wall unit (1.0) the two overlapped, and the
            # joined placeholder became self-intersecting - which made every
            # attempt to place the run invalid, so a kitchen could finish with
            # no counter at all.
            top_lo = max(
                z - self.cabinet_top_height - 0.1,
                self.cabinet_bottom_height + 0.14,
            )
            # Held a few mm forward of the base box's back face. The 1.08
            # above used to keep the two boxes from sharing a plane; with the
            # placeholder trimmed to the real depth they became coplanar, and a
            # joined box with a doubled back face has no single tagged back for
            # stable_against to seat on - every placement attempt was rejected.
            box_top = new_bbox(-x / 2 + 0.005, x * 0.16, 0, y, top_lo, z)
            box = butil.join_objects([box, box_top])

        return box

    def create_asset(self, **params):
        x, y, z = self.dimensions
        parts = []

        cabinet_bottom_height = self.cabinet_bottom_height
        cabinet_top_height = self.cabinet_top_height

        cabinet_bottom_factory = KitchenCabinetFactory(
            self.factory_seed,
            dimensions=(x, y - 0.15, cabinet_bottom_height),
            drawer_only=True,
        )
        cabinet_bottom = cabinet_bottom_factory(i=0)
        parts.append(cabinet_bottom)

        surface.add_geomod(cabinet_bottom, geometry_nodes_add_cabinet_top, apply=True)

        if not self.island:
            # top
            top_mid_width = uniform(1.0, 1.3)
            cabinet_top_width = (y - top_mid_width) / 2.0 - 0.05

            cabinet_top_factory = KitchenCabinetFactory(
                self.factory_seed,
                dimensions=(x / 2.0, cabinet_top_width, cabinet_top_height),
                drawer_only=False,
            )
            cabinet_top_left = cabinet_top_factory(i=0)
            cabinet_top_right = cabinet_top_factory(i=1)

            cabinet_top_left.location = (-x / 4.0, 0.0, z - cabinet_top_height)
            cabinet_top_right.location = (
                -x / 4.0,
                y - cabinet_top_width,
                z - cabinet_top_height,
            )
            bpy.context.view_layer.update()

            # Wall units grow past the requested width (side boards, doors), so
            # the real gap is smaller than top_mid_width + 0.1. Sizing the hood
            # from the requested gap is what drove it into the left carcass.
            left_max_y = max(
                (cabinet_top_left.matrix_world @ Vector(c)).y
                for c in cabinet_top_left.bound_box
            )
            right_min_y = min(
                (cabinet_top_right.matrix_world @ Vector(c)).y
                for c in cabinet_top_right.bound_box
            )
            gap = right_min_y - left_max_y
            clearance = 0.03
            hood_width = gap - 2 * clearance
            if hood_width < 0.45:
                hood_width = max(0.35, gap - 0.016)
            hood_y = (left_max_y + right_min_y) / 2.0

            range_hood_factory = RangeHoodFactory(
                self.factory_seed,
                dimensions=(x * 0.66, hood_width, cabinet_top_height),
            )
            top_mid = range_hood_factory(i=0)
            range_hood_factory.finalize_assets([top_mid])
            # The hood mesh runs from its own origin forward in +x, so placing
            # the origin on the wall line (-x/2) seats its back against the
            # wall and its canopy over the hob.
            top_mid.location = (-x * 0.5, hood_y, z - cabinet_top_height + 0.05)
            bpy.context.view_layer.update()
            hood_ys = [
                (top_mid.matrix_world @ Vector(c)).y for c in top_mid.bound_box
            ]
            hy0, hy1 = min(hood_ys), max(hood_ys)
            overlap_l = (left_max_y + clearance) - hy0
            overlap_r = hy1 - (right_min_y - clearance)
            span = hy1 - hy0
            trim = max(0.0, overlap_l) + max(0.0, overlap_r)
            if trim > 1e-4 and span > 0.3:
                top_mid.scale.y *= max(0.35, span - trim) / span
                top_mid.location.y = hood_y
                butil.apply_transform(top_mid)

            hob = CooktopFactory(self.factory_seed).spawn_asset(0)
            hob.location = (0.0, y / 2.0, cabinet_bottom_height + 0.05)

            parts += [cabinet_top_left, cabinet_top_right, top_mid, hob]

        kitchen_space = butil.join_objects(
            parts
        )  # [cabinet_bottom, sink, cabinet_top_left, cabinet_top_right, top_mid])

        if not self.island:
            kitchen_space.dimensions = self.dimensions
        butil.apply_transform(kitchen_space)

        if not self.island:
            # Seat the back of the run on the wall line. `dimensions` scales
            # about the origin, not the bounding box, so the fit above leaves
            # the carcass a centimetre or so shy of -x/2 - which reads as the
            # units standing off the plaster.
            corners = [kitchen_space.matrix_world @ Vector(v) for v in kitchen_space.bound_box]
            kitchen_space.location.x += -x / 2 - min(c.x for c in corners)
            butil.apply_transform(kitchen_space)

        tagging.tag_system.relabel_obj(kitchen_space)

        return kitchen_space


class KitchenIslandFactory(KitchenSpaceFactory):
    def __init__(self, factory_seed):
        super(KitchenIslandFactory, self).__init__(
            factory_seed=factory_seed,
            island=True,
        )
