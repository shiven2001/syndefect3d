# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file
# in the root directory of this source tree.

"""Procedural Type-G wall socket (common in HK / UK handover flats)."""

import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.wall_decorations.primitives import (
    assign,
    box,
    shade_smooth,
    solid_material,
)
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed


class WallPlugFactory(AssetFactory):
    """1- or 2-gang switched Type-G socket on a plastic faceplate."""

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse=coarse)
        with FixedSeed(factory_seed):
            self.n_gangs = 1 if uniform() < 0.45 else 2
            self.plate_h = uniform(0.082, 0.090)
            self.gang_w = uniform(0.080, 0.088)
            self.gap = uniform(0.004, 0.008)
            self.plate_w = (
                self.n_gangs * self.gang_w + (self.n_gangs - 1) * self.gap + 0.012
            )
            self.thickness = uniform(0.008, 0.012)
            self.has_switch = uniform() < 0.85
            tone = uniform(0.86, 0.97)
            self.plate_color = (tone, tone * uniform(0.98, 1.0), tone * uniform(0.94, 1.0))
            self.slot_color = (0.04, 0.04, 0.04)
            switch_v = tone * uniform(0.92, 1.02)
            self.switch_color = (switch_v, switch_v, switch_v * 0.98)

    def create_placeholder(self, **params):
        return new_bbox(
            -0.002,
            self.thickness + 0.004,
            -self.plate_w / 2,
            self.plate_w / 2,
            -self.plate_h / 2,
            self.plate_h / 2,
        )

    def _gang_center_y(self, i):
        if self.n_gangs == 1:
            return 0.0
        span = self.gang_w + self.gap
        return (i - 0.5) * span

    def create_asset(self, **params):
        plate_mat = solid_material(
            f"WallPlugPlate_{self.factory_seed}",
            self.plate_color,
            roughness=uniform(0.28, 0.45),
        )
        slot_mat = solid_material(
            f"WallPlugSlot_{self.factory_seed}",
            self.slot_color,
            roughness=0.55,
        )
        switch_mat = solid_material(
            f"WallPlugSwitch_{self.factory_seed}",
            self.switch_color,
            roughness=0.32,
        )

        plate = box(
            (self.thickness, self.plate_w, self.plate_h),
            location=(self.thickness / 2, 0, 0),
            name="wallplug_plate",
        )
        assign(plate, plate_mat)
        butil.modify_mesh(plate, "BEVEL", width=0.0012, segments=2)
        shade_smooth(plate)
        parts = [plate]

        well_t = 0.002
        for i in range(self.n_gangs):
            cy = self._gang_center_y(i)
            well = box(
                (well_t, self.gang_w * 0.78, self.plate_h * 0.72),
                location=(self.thickness + well_t / 2 - 0.001, cy, -0.002),
                name=f"wallplug_well_{i}",
            )
            assign(well, plate_mat)
            parts.append(well)

            if self.has_switch:
                sw = box(
                    (0.004, self.gang_w * 0.42, 0.014),
                    location=(
                        self.thickness + 0.003,
                        cy,
                        self.plate_h * 0.22,
                    ),
                    name=f"wallplug_switch_{i}",
                )
                assign(sw, switch_mat)
                parts.append(sw)

            slot_z = -0.008 if self.has_switch else 0.004
            slot_x = self.thickness + 0.002
            # Type-G: two angled upper pins + one lower earth.
            for sx, sy, sz, ly, lz in (
                (0.003, 0.005, 0.013, -0.011, slot_z + 0.010),
                (0.003, 0.005, 0.013, 0.011, slot_z + 0.010),
                (0.003, 0.006, 0.016, 0.0, slot_z - 0.008),
            ):
                slot = box(
                    (sx, sy, sz),
                    location=(slot_x, cy + ly, lz),
                    name="wallplug_slot",
                )
                assign(slot, slot_mat)
                parts.append(slot)

        for sign in (-1, 1):
            screw = box(
                (0.002, 0.006, 0.006),
                location=(
                    self.thickness + 0.001,
                    0.0,
                    sign * (self.plate_h / 2 - 0.008),
                ),
                name="wallplug_screw",
            )
            assign(screw, slot_mat)
            parts.append(screw)

        obj = join_objects(parts)
        obj.name = f"WallPlugFactory({self.factory_seed}).socket"
        return obj
