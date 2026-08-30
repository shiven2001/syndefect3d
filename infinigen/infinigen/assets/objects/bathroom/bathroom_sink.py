# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import bmesh

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.bathroom.bathtub import BathtubFactory
from infinigen.assets.objects.bathroom.fittings import bathroom_chrome
from infinigen.assets.materials.wood.wood import InteriorWood
from infinigen.assets.objects.table_decorations import TapFactory
from infinigen.assets.utils.decorate import (
    read_co,
    subdivide_edge_ring,
    subsurf,
    write_attribute,
)
from infinigen.assets.utils.object import (
    join_objects,
    new_base_cylinder,
    new_bbox,
    new_cube,
    new_cylinder,
)
from infinigen.core import surface
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform, weighted_sample



def _vanity_wood():
    """Stained interior wood for the under-sink cabinet (not white laminate)."""
    return InteriorWood()


class BathroomSinkFactory(BathtubFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BathroomSinkFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.width = uniform(0.6, 0.9)
            self.size = self.width * log_uniform(0.55, 0.8)
            self.depth = self.width * log_uniform(0.2, 0.4)
            self.contour_fn = self.make_box_contour
            self.sink_types = np.random.choice(["undermount", "drop-in", "vessel"])
            self.has_stand = False
            match self.sink_types:
                case "undermount":
                    self.bathtub_type = "freestanding"
                    self.has_extrude = uniform() < 0.7
                case "drop-in":
                    self.bathtub_type = "alcove"
                    self.has_extrude = True
                case _:
                    self.bathtub_type = np.random.choice(["alcove", "freestanding"])
                    self.has_extrude = uniform() < 0.7
                    self.has_stand = True
            self.tap_factory = TapFactory(self.factory_seed)

            self.disp_x = [self.disp_x[0], self.disp_x[0]]
            self.alcove_levels = 0 if uniform() < 0.5 else np.random.randint(2, 4)
            self.thickness = 0.01 if self.has_base else uniform(0.01, 0.03)
            self.size_extrude = uniform(0.2, 0.35)
            self.tap_offset = uniform(0.0, 0.05)
            # A pedestal is roughly a third the width of the basin. At the old
            # 0.15-0.2 it came out about 80 mm across under a 900 mm basin, which
            # is what made the stand look spindly.
            self.stand_radius = self.width / 2 * log_uniform(0.34, 0.46)
            self.stand_bottom = self.stand_radius * uniform(1.15, 1.4)
            self.stand_height = uniform(0.80, 0.88) - self.depth
            self.is_stand_circular = uniform() < 0.5
            self.is_hole_centered = True

            surface_gen_class = weighted_sample(material_assignments.ceramics)
            self.surface_material_gen = surface_gen_class

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -(self.size_extrude + 1) * self.size,
            0,
            0,
            self.width,
            -self.stand_height if self.has_stand else 0,
            self.depth,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        self.surface = self.surface_material_gen()
        if self.has_base:
            obj = self.make_base()
            cutter = self.make_cutter()
            butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
        else:
            obj = self.make_bowl()
            self.remove_top(obj)
            butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness)
            subsurf(obj, self.side_levels)
        obj.location = np.array(obj.location) - np.min(read_co(obj), 0)
        butil.apply_transform(obj, True)
        obj.scale = np.array([self.width, self.size, self.depth]) / np.array(
            obj.dimensions
        )
        butil.apply_transform(obj, True)
        if self.has_extrude:
            self.extrude_back(obj)
        if self.has_stand:
            obj, handles = self.add_stand(obj)
        else:
            handles = []
        hole = self.add_hole(obj)
        obj = join_objects([obj, hole])
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj, True)
        chrome_children = []
        if self.has_extrude:
            tap = self.tap_factory(np.random.randint(1e7))
            min_x = np.min(read_co(tap)[:, 0])
            tap.location = (
                (-1 - self.size_extrude + self.tap_offset) * self.size - min_x,
                self.width / 2,
                self.depth,
            )
            butil.apply_transform(tap, True)
            chrome_children.append(tap)
        # Handles are built in pre-rotation space; match the sink's 90° turn.
        for handle in handles:
            handle.rotation_euler[-1] = np.pi / 2
            butil.apply_transform(handle, True)
            chrome_children.append(handle)
        chrome = bathroom_chrome()
        for child in chrome_children:
            surface.assign_material(child, chrome)
            child.parent = obj
        self._chrome_children = chrome_children
        return obj

    def extrude_back(self, obj):
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="FACE")
            bpy.ops.mesh.select_all(action="DESELECT")
            bm = bmesh.from_edit_mesh(obj.data)
            for f in bm.faces:
                f.select_set(
                    f.calc_center_median()[1] > self.size / 2 and f.normal[1] > 0.1
                )
            bm.select_flush(False)
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": (0, self.size_extrude * self.size, 0)}
            )

    def add_stand(self, obj):
        """Basin sits on a vanity cabinet rather than a pedestal.

        The old tapered column read as a narrow post under a wide basin, and
        sat forward of centre. A cabinet fills the basin footprint, so it is
        flush with the wall behind and carries the bowl properly.

        Faces are tagged because finalize_assets repaints the whole joined
        object with ceramic using clear=True. Bar pulls stay separate so they
        keep bathroom chrome instead of being painted porcelain.
        """
        w = self.width
        # Include the rear deck (extrude_back) so the carcass matches the basin.
        d = self.size * (1 + (self.size_extrude if self.has_extrude else 0.0))
        h = self.stand_height
        kick = min(0.075, h * 0.12)
        cx, cy = w / 2, d / 2

        carcass_h = h - kick
        carcass = new_cube()
        carcass.scale = w / 2, d / 2, carcass_h / 2
        carcass.location = cx, cy, -kick - carcass_h / 2
        butil.apply_transform(carcass, True)
        butil.modify_mesh(carcass, "BEVEL", width=0.004, segments=2)
        write_attribute(carcass, 1, "vanity", "FACE")
        parts = [carcass]

        # Recessed plinth, so the cabinet does not look like a solid block.
        plinth = new_cube()
        plinth.scale = (w - 0.05) / 2, (d - 0.04) / 2, kick / 2
        plinth.location = cx, cy, -kick / 2
        butil.apply_transform(plinth, True)
        write_attribute(plinth, 1, "vanity", "FACE")
        parts.append(plinth)

        # Two door fronts with a shadow gap between them.
        door_w = (w - 0.03) / 2
        handles = []
        for sign in (-1, 1):
            door = new_cube()
            door.scale = door_w / 2, 0.008, (carcass_h - 0.02) / 2
            door.location = (
                cx + sign * (door_w / 2 + 0.0075),
                cy - d / 2 - 0.006,
                -kick - carcass_h / 2,
            )
            butil.apply_transform(door, True)
            butil.modify_mesh(door, "BEVEL", width=0.002, segments=2)
            write_attribute(door, 1, "vanity", "FACE")
            parts.append(door)

            pull = new_cylinder(vertices=12)
            pull.scale = 0.006, 0.006, door_w * 0.34
            pull.rotation_euler = (0, np.pi / 2, 0)
            pull.location = (
                cx + sign * (door_w / 2 + 0.0075),
                cy - d / 2 - 0.022,
                -kick - 0.05,
            )
            butil.apply_transform(pull, True)
            handles.append(pull)

        return join_objects([obj] + parts), handles

    def finalize_assets(self, assets):
        self.surface.apply(assets, clear=True)
        _vanity_wood().apply(assets, selection="vanity")
        chrome = bathroom_chrome()
        for child in getattr(self, "_chrome_children", []):
            surface.assign_material(child, chrome)
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)


class StandingSinkFactory(BathroomSinkFactory):
    def __init__(self, factory_seed, coarse=False):
        super(StandingSinkFactory, self).__init__(factory_seed, coarse)
        self.bathtub_type = "freestanding"
        self.has_extrude = True
        self.has_stand = True
